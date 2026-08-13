import re
import logging
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from sentence_transformers import CrossEncoder

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Флаги для включения продвинутых техник
USE_QUERY_REWRITING = True
USE_HYDE = True
USE_MULTI_QUERY = True
USE_RERANKING = True
USE_JUDGE = True
USE_DSPY = True
TOP_K_BASE = 20
TOP_K_FINAL = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 1. Подключение к локальной LLM
llm = ChatOpenAI(
    api_key="none",
    base_url="http://localhost:1234/v1/",
    model="qwen/qwen3.5-9b",
    temperature=0.1,
)

# 2. Загрузка документа
source = "test_document.txt"
if source.endswith('.pdf'):
    loader = PyPDFLoader(source)
elif source.endswith('.docx'):
    loader = Docx2txtLoader(source)
elif source.endswith('.txt'):
    loader = TextLoader(source, encoding='utf-8')
elif source.startswith('http'):
    loader = WebBaseLoader(source)
else:
    raise ValueError("Поддерживаются: PDF, DOCX, TXT, URL")
documents = loader.load()
logger.info(f"Загружено {len(documents)} страниц")

# 3. Нарезка на чанки с перекрытием
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
# Добавляем уникальный идентификатор для каждого чанка для RRF
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = f"chunk_{i}"
logger.info(f"Создано {len(chunks)} чанков")

# 4. Векторное представление и индекс FAISS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_BASE})
logger.info("Векторный индекс создан")

# 5. Query Rewriting
rewrite_prompt = ChatPromptTemplate.from_template(
    "Перепиши следующий вопрос пользователя так, чтобы он содержал ключевые термины "
    "для поиска в базе знаний. Сохрани смысл, но сделай запрос более формальным и точным.\n"
    "Исходный вопрос: {question}\nПереписанный вопрос:"
)
rewriter_chain = rewrite_prompt | llm | StrOutputParser()

# 6. HyDE
hyde_prompt = ChatPromptTemplate.from_template(
    "Напиши гипотетический ответ на вопрос пользователя. Ответ должен быть похож на "
    "фрагмент из документа, содержащий все факты, которые могли бы быть в ответе.\n"
    "Вопрос: {question}\nГипотетический ответ:"
)
hyde_chain = hyde_prompt | llm | StrOutputParser()

def generate_hyde_query(question: str) -> Optional[str]:
    """Генерирует гипотетический ответ, возвращает строку или None при ошибке"""
    try:
        return hyde_chain.invoke({"question": question})
    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        return None

# 7. Multi-Query и RRF-слияние
multi_query_prompt = ChatPromptTemplate.from_template(
    "Сгенерируй 3 разных варианта поискового запроса, которые помогут найти "
    "информацию по следующему вопросу. Каждый вариант должен быть кратким и точным.\n"
    "Выдай только список запросов, по одному на строку, без нумерации и пояснений.\n"
    "Вопрос: {question}\nВарианты:"
)

def generate_queries(question: str) -> List[str]:
    """Генерирует список поисковых запросов с защитой от ошибок парсинга"""
    try:
        response = llm.invoke(multi_query_prompt.format(question=question))
        lines = response.content.strip().split('\n')
        queries = []
        for line in lines:
            line = line.strip()
            # Убираем возможные маркеры вроде цифр, дефисов или звёздочек
            line = re.sub(r'^[\d\-*•]+\.?\s*', '', line)
            if line:
                queries.append(line)
        return queries if queries else [question]
    except Exception as e:
        logger.error(f"Multi-query generation failed: {e}")
        return [question]

def reciprocal_rank_fusion(results_lists: List[List[Document]], k: int = 60) -> List[Document]:
    """Объединяет результаты поиска по RRF, использует chunk_id как ключ"""
    scores = {}
    doc_map = {}
    for docs in results_lists:
        for rank, doc in enumerate(docs, 1):
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:100])
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]

# 8. Reranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query: str, documents: List[Document], top_k: int = TOP_K_FINAL) -> List[Document]:
    """Переранжирует документы с помощью cross-encoder"""
    if not documents:
        return []
    try:
        pairs = [[query, doc.page_content] for doc in documents]
        scores = cross_encoder.predict(pairs)
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [documents[i] for i in sorted_idx[:top_k]]
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return documents[:top_k]

# 9. Кастомный ретривер, объединяющий все техники
class AdvancedRetriever(BaseRetriever):
    base_retriever: object = Field(exclude=True)
    llm: object = Field(exclude=True)
    cross_encoder: object = Field(exclude=True)
    top_k_final: int = TOP_K_FINAL

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Переписывание запроса
        if USE_QUERY_REWRITING:
            try:
                rewritten = rewriter_chain.invoke({"question": query})
                logger.info(f"Rewritten: {rewritten}")
            except Exception as e:
                logger.error(f"Rewrite failed: {e}")
                rewritten = query
        else:
            rewritten = query

        # Формирование списка поисковых запросов
        search_queries = [rewritten]

        # HyDE как отдельный запрос на основе исходного вопроса
        if USE_HYDE:
            hyde_query = generate_hyde_query(query)
            if hyde_query:
                search_queries.append(hyde_query)
                logger.info(f"HyDE query added with length {len(hyde_query)}")

        # Мультизапросы на основе исходного вопроса
        if USE_MULTI_QUERY:
            extra_queries = generate_queries(query)
            search_queries.extend(extra_queries)
            logger.info(f"Multi-query added: {len(extra_queries)} variants")

        # Удаление дубликатов запросов
        unique_queries = list(dict.fromkeys(search_queries))
        logger.info(f"Total unique search queries: {len(unique_queries)}")

        # Поиск по каждому запросу
        all_results = []
        for q in unique_queries:
            try:
                docs = self.base_retriever.invoke(q)
                all_results.append(docs)
            except Exception as e:
                logger.error(f"Search failed for query '{q[:50]}...': {e}")
                all_results.append([])

        # Слияние результатов через RRF
        merged = reciprocal_rank_fusion(all_results, k=60)

        # Переранжирование
        if USE_RERANKING:
            reranked = rerank_documents(query, merged, self.top_k_final)
        else:
            reranked = merged[:self.top_k_final]
        return reranked
advanced_retriever = AdvancedRetriever(
    base_retriever=base_retriever, llm=llm, cross_encoder=cross_encoder
)

# 10. Память диалога
chat_history = InMemoryChatMessageHistory()

# 11. Промпт для генерации ответа с учётом истории
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Ты - полезный ассистент. Отвечай на вопрос, используя только информацию из предоставленного контекста. Если ответа нет в контексте, скажи: 'Я не знаю, в документах этого нет'."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Контекст:\n{context}\n\nВопрос: {question}")
])

# 12. Сборка основной RAG-цепочки
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=advanced_retriever,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    return_source_documents=True,
    verbose=False
)

# 13. LLM‑as‑a‑Judge с улучшенным парсингом
if USE_JUDGE:
    faith_prompt = ChatPromptTemplate.from_template(
        "Оцени, насколько ответ соответствует предоставленному контексту, по шкале от 0 до 1, "
        "где 0 — ответ полностью выдуман, 1 — ответ полностью основан на контексте. "
        "Сначала кратко объясни причину, затем выведи только число.\n"
        "Вопрос: {question}\nКонтекст: {context}\nОтвет: {answer}\nОценка только число:"
    )
    rel_prompt = ChatPromptTemplate.from_template(
        "Оцени, насколько ответ релевантен вопросу, по шкале 0–1, где 0 — не отвечает, 1 — полностью отвечает. "
        "Сначала кратко объясни причину, затем выведи только число.\n"
        "Вопрос: {question}\nОтвет: {answer}\nОценка только число:"
    )

    def extract_number(text: str) -> float:
        """Извлекает первое число из строки, возвращает 0.0 если не найдено"""
        match = re.search(r'(\d+\.?\d*)', text)
        if match:
            return float(match.group(1))
        return 0.0

    def judge_faithfulness(question, answer, context):
        try:
            chain = faith_prompt | llm | StrOutputParser()
            raw = chain.invoke({"question": question, "context": context, "answer": answer})
            return extract_number(raw)
        except Exception as e:
            logger.error(f"Faithfulness judge failed: {e}")
            return 0.0

    def judge_relevancy(question, answer):
        try:
            chain = rel_prompt | llm | StrOutputParser()
            raw = chain.invoke({"question": question, "answer": answer})
            return extract_number(raw)
        except Exception as e:
            logger.error(f"Relevancy judge failed: {e}")
            return 0.0

# 14. DSPy с расширенным датасетом
if USE_DSPY:
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    lm_dspy = dspy.LM('openai/qwen/qwen3.5-9b', api_base="http://localhost:1234/v1/", api_key="none")
    dspy.settings.configure(lm=lm_dspy)
    class AnswerGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought("question, context -> answer")
        def forward(self, question, context):
            return self.generate(question=question, context=context)
    trainset = [
        dspy.Example(question="Какие продукты у компании?",
                     context="Ключевые продукты: 1. RAG-платформа «Знание»; 2. Ассистент «Умник»; 3. Аналитическая панель «Инсайт».",
                     answer="RAG-платформа «Знание», Ассистент «Умник», Аналитическая панель «Инсайт»"),
        dspy.Example(question="Когда основана компания?",
                     context="ООО «Интеллектуальные Системы» основано в 2020 году.",
                     answer="Компания основана в 2020 году."),
        dspy.Example(question="Сколько сотрудников планируется в 2026 году?",
                     context="Планы на 2026 год: увеличение штата до 80 человек.",
                     answer="В 2026 году планируется увеличить штат до 80 человек."),
        dspy.Example(question="Какие векторные БД используются?",
                     context="Технологический стек: векторные БД - FAISS, Qdrant, Pinecone.",
                     answer="Используются FAISS, Qdrant, Pinecone."),
        dspy.Example(question="Какие модели применяются?",
                     context="Модели: открытые (Llama, Mistral, Qwen) и проприетарные (OpenAI, Anthropic).",
                     answer="Применяются открытые модели Llama, Mistral, Qwen, а также проприетарные OpenAI и Anthropic."),
    ]
    trainset = [x.with_inputs('question', 'context') for x in trainset]
    def validate(example, pred, trace=None):
        return any(word in pred.answer.lower() for word in example.answer.lower().split())
    optimizer = BootstrapFewShot(metric=validate, max_bootstrapped_demos=3)
    optimized_generator = optimizer.compile(AnswerGenerator(), trainset=trainset)

# 15. Интерактивный цикл с корректным обновлением истории
print("Чат-бот с продвинутыми техниками готов. Введите 'exit' для выхода.")
while True:
    user_input = input("Вы: ")
    if user_input.lower() in ["exit", "quit"]:
        print("До свидания.")
        break
    result = qa_chain.invoke({"question": user_input, "chat_history": chat_history.messages})
    answer = result["answer"]
    source_docs = result.get("source_documents", [])
    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(answer)
    print(f"Бот: {answer}")
    if USE_JUDGE:
        context_text = "\n".join([doc.page_content for doc in source_docs])
        faith = judge_faithfulness(user_input, answer, context_text)
        rel = judge_relevancy(user_input, answer)
        print(f"[Faithfulness: {faith:.2f}, Relevancy: {rel:.2f}]")
    print()