from typing import List
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from sentence_transformers import CrossEncoder

# -------------------- Флаги для включения продвинутых техник --------------------
USE_QUERY_REWRITING = True      # Переписывать запрос с помощью LLM
USE_HYDE = False                # Генерировать гипотетический ответ
USE_MULTI_QUERY = True          # Генерировать несколько вариантов запроса
USE_RERANKING = True            # Переранжировать результаты через Cross-Encoder
USE_JUDGE = True                # Оценивать Faithfulness и Relevancy ответа
USE_DSPY = False                # Использовать DSPy для оптимизации промптов
TOP_K_BASE = 20                 # Сколько документов достаём на первом этапе
TOP_K_FINAL = 4                 # Сколько отдаём в генерацию после всех улучшений
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# -------------------- 1. Подключение к локальной LLM --------------------
llm = ChatOpenAI(
    api_key="none",
    base_url="http://localhost:1234/v1/",
    model="qwen/qwen3.5-9b",
    temperature=0.1,
)

# -------------------- 2. Загрузка документа --------------------
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
print(f"Загружено {len(documents)} страниц")

# -------------------- 3. Нарезка на чанки с перекрытием --------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Создано {len(chunks)} чанков")

# -------------------- 4. Векторное представление и индекс FAISS --------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_BASE})
print("Векторный индекс создан")

# -------------------- 5. Query Rewriting --------------------
rewrite_prompt = ChatPromptTemplate.from_template(
    "Перепиши следующий вопрос пользователя так, чтобы он содержал ключевые термины "
    "для поиска в базе знаний. Сохрани смысл, но сделай запрос более формальным и точным.\n"
    "Исходный вопрос: {question}\nПереписанный вопрос:"
)
rewriter_chain = rewrite_prompt | llm | StrOutputParser()

# -------------------- 6. HyDE --------------------
hyde_prompt = ChatPromptTemplate.from_template(
    "Напиши гипотетический ответ на вопрос пользователя. Ответ должен быть похож на "
    "фрагмент из документа, содержащий все факты, которые могли бы быть в ответе.\n"
    "Вопрос: {question}\nГипотетический ответ:"
)
hyde_chain = hyde_prompt | llm | StrOutputParser()
def enhance_query(question: str) -> str:
    if USE_HYDE:
        return hyde_chain.invoke({"question": question})
    elif USE_QUERY_REWRITING:
        return rewriter_chain.invoke({"question": question})
    else:
        return question

# -------------------- 7. Multi-Query и RRF-слияние --------------------
multi_query_prompt = ChatPromptTemplate.from_template(
    "Сгенерируй 3 разных варианта поискового запроса, которые помогут найти "
    "информацию по следующему вопросу. Каждый вариант должен быть кратким и точным.\n"
    "Вопрос: {question}\nВарианты:"
)

def generate_queries(question: str) -> List[str]:
    response = llm.invoke(multi_query_prompt.format(question=question))
    queries = response.content.strip().split('\n')
    return [q.strip() for q in queries if q.strip()]

def reciprocal_rank_fusion(results_lists: List[List], k: int = 60) -> List:
    scores = {}
    for docs in results_lists:
        for rank, doc in enumerate(docs, 1):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank)
    sorted_contents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for content, _ in sorted_contents:
        for doc in results_lists[0]:
            if doc.page_content == content and doc not in result:
                result.append(doc)
                break
    return result

# -------------------- 8. Reranking --------------------
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
def rerank_documents(query: str, documents: List, top_k: int = TOP_K_FINAL) -> List:
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [documents[i] for i in sorted_idx[:top_k]]

# -------------------- 9. Кастомный ретривер, объединяющий все техники --------------------
class AdvancedRetriever(BaseRetriever):
    base_retriever: object = Field(exclude=True)
    llm: object = Field(exclude=True)
    cross_encoder: object = Field(exclude=True)
    top_k_final: int = TOP_K_FINAL

    def _get_relevant_documents(self, query: str) -> List[Document]:
        enhanced = enhance_query(query)
        if USE_MULTI_QUERY:
            queries = generate_queries(enhanced)
            all_docs = []
            for q in queries:
                docs = self.base_retriever.get_relevant_documents(q)
                all_docs.append(docs)
            merged = reciprocal_rank_fusion(all_docs, k=60)
        else:
            merged = self.base_retriever.get_relevant_documents(enhanced)
        if USE_RERANKING:
            reranked = rerank_documents(query, merged, self.top_k_final)
        else:
            reranked = merged[:self.top_k_final]
        return reranked
advanced_retriever = AdvancedRetriever(
    base_retriever=base_retriever, llm=llm, cross_encoder=cross_encoder
)

# -------------------- 10. Память диалога --------------------
chat_history = InMemoryChatMessageHistory()

# -------------------- 11. Промпт для генерации ответа --------------------
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Ты - полезный ассистент. Отвечай на вопрос, используя только информацию из предоставленного контекста. Если ответа нет в контексте, скажи: 'Я не знаю, в документах этого нет'."),
    ("human", "Контекст:\n{context}\n\nВопрос: {question}")
])

# -------------------- 12. Сборка основной RAG-цепочки --------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=advanced_retriever,
    chat_history=chat_history,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    return_source_documents=True,
    verbose=False
)

# -------------------- 13. LLM‑as‑a‑Judge --------------------
if USE_JUDGE:
    faith_prompt = ChatPromptTemplate.from_template(
        "Оцени, насколько ответ соответствует предоставленному контексту, по шкале от 0 до 1, "
        "где 0 — ответ полностью выдуман, 1 — ответ полностью основан на контексте.\n"
        "Вопрос: {question}\nКонтекст: {context}\nОтвет: {answer}\nОценка:"
    )
    rel_prompt = ChatPromptTemplate.from_template(
        "Оцени, насколько ответ релевантен вопросу, по шкале 0–1, где 0 — не отвечает, 1 — полностью отвечает.\n"
        "Вопрос: {question}\nОтвет: {answer}\nОценка:"
    )

    def judge_faithfulness(question, answer, context):
        chain = faith_prompt | llm | StrOutputParser()
        score_str = chain.invoke({"question": question, "context": context, "answer": answer})
        try:
            return float(score_str.strip())
        except:
            return 0.0

    def judge_relevancy(question, answer):
        chain = rel_prompt | llm | StrOutputParser()
        score_str = chain.invoke({"question": question, "answer": answer})
        try:
            return float(score_str.strip())
        except:
            return 0.0

# -------------------- 14. Опционально: DSPy --------------------
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
                     context="Ключевые продукты: 1. RAG-платформа «Знание» ...",
                     answer="RAG-платформа «Знание», Ассистент «Умник», Аналитическая панель «Инсайт»"),
    ]
    trainset = [x.with_inputs('question', 'context') for x in trainset]
    def validate(example, pred, trace=None):
        return example.answer.lower() in pred.answer.lower()
    optimizer = BootstrapFewShot(metric=validate, max_bootstrapped_demos=2)
    optimized_generator = optimizer.compile(AnswerGenerator(), trainset=trainset)

# -------------------- 15. Интерактивный цикл --------------------
print("Чат-бот с продвинутыми техниками готов. Введите 'exit' для выхода.")
while True:
    user_input = input("Вы: ")
    if user_input.lower() in ["exit", "quit"]:
        print("До свидания.")
        break
    result = qa_chain.invoke({"question": user_input})
    answer = result["answer"]
    source_docs = result.get("source_documents", [])
    context_text = "\n".join([doc.page_content for doc in source_docs])
    print(f"Бот: {answer}")
    if USE_JUDGE:
        faith = judge_faithfulness(user_input, answer, context_text)
        rel = judge_relevancy(user_input, answer)
        print(f"[Faithfulness: {faith:.2f}, Relevancy: {rel:.2f}]")
    print()