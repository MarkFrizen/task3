from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

# Подключение к LLM
# Используем локальный сервер LM Studio с моделью Qwen 9B
# Модель применяется для переписывания запросов, генерации гипотетических ответов и формирования финального ответа
llm = ChatOpenAI(
    api_key="none",
    base_url="http://192.168.8.11:1234/v1/",
    model="qwen/qwen3.5-9b",
    temperature=0.1,
)

# Модель для Rerank
# Кросс‑энкодер оценивает релевантность пары запрос–документ
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Загрузка документа
# Поддерживаются PDF, DOCX, TXT, веб‑страницы
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
    raise ValueError("Поддерживаются PDF, DOCX, TXT, URL")
documents = loader.load()
print(f"Загружено {len(documents)} страниц")

# Умная нарезка текста с перекрытием
# Текст делится на чанки с перекрытием, чтобы не терять смысл на границах
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Создано {len(chunks)} чанков")

# Создание векторного индекса
# Преобразуем текст в векторы с мультиязычной моделью и сохраняем в FAISS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()
print("Индекс создан")

# История диалога
# Сохраняет контекст беседы для учёта предыдущих вопросов и ответов
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Промпт для переписывания запроса
# LLM создаёт 3 семантически разных варианта запроса
rewrite_prompt = ChatPromptTemplate.from_messages([
    "Ты перефразируешь запрос пользователя в 3 семантически разных варианта. Каждый вариант — одна короткая строка. Не нумеруй и не добавляй пояснений.",
    "{query}"
])
rewrite_chain = rewrite_prompt.compile(llm)

def rewrite_query(query, n_variants=3):
    variants = rewrite_chain.invoke({"query": query}).content.splitlines()
    variants = [v.strip() for v in variants if v.strip()]
    while len(variants) < n_variants:
        variants.append(query)
    return variants[:n_variants]

# Промпт для HyDE
# Генерируем гипотетический ответ, чтобы искать документы по его эмбеддингу
hyde_prompt = ChatPromptTemplate.from_messages([
    "Напиши краткий, но содержательный ответ на вопрос, будто у тебя есть доступ к полной базе знаний. Не использу фразы вроде я не знаю. Только 2–4 предложения.",
    "{query}"
])
hyde_chain = hyde_prompt.compile(llm)
def retrieve_with_hyde(query, retriever, k=5):
    hypothetical_answer = hyde_chain.invoke({"query": query}).content
    return retriever.get_relevant_documents(hypothetical_answer, k=k)

# Multi‑Query Retrieval
# Поиск по нескольким перефразированным запросам, объединение и дедупликация результатов
def multi_query_retrieve(query, retriever, n_variants=3, k_per_variant=5):
    variants = rewrite_query(query, n_variants)
    docs = []
    for v in variants:
        docs.extend(retriever.get_relevant_documents(v, k=k_per_variant))
    seen = set()
    unique_docs = []
    for d in docs:
        key = d.page_content
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)
    return unique_docs

# Rerank с кросс‑энкодером
# Пересчитывает релевантность документов и возвращает топ‑N
def rerank_documents(query, docs, top_n=5):
    if len(docs) == 0:
        return docs
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, s in ranked[:top_n]]

# Продвинутый ретривер
# Объединяет Multi‑Query, HyDE и Rerank для более точного поиска
def advanced_retriever(query, base_retriever, top_k=5):
    multi_docs = multi_query_retrieve(query, base_retriever, n_variants=3, k_per_variant=6)
    hyde_docs = retrieve_with_hyde(query, base_retriever, k=6)
    all_docs = multi_docs + hyde_docs
    seen = set()
    combined = []
    for d in all_docs:
        key = d.page_content
        if key not in seen:
            seen.add(key)
            combined.append(d)
    reranked = rerank_documents(query, combined, top_n=top_k)
    return reranked

# Обертка для совместимости с LangChain
def retriever_wrapper(query):
    return advanced_retriever(query, retriever, top_k=4)

# Промпт для финального ответа
# Модель отвечает строго по контексту, если не знает — признаётся
prompt_template = ChatPromptTemplate.from_messages([
    "Ты полезный ассистент. Отвечай на вопрос, используя только информацию из предоставленного контекста. Если ответа нет в контексте, скажи Я не знаю, в документах этого нет",
    "Контекст\n{context}\n\nВопрос\n{question}"
])

# Сборка RAG‑цепочки
# Объединяет LLM, продвинутый ретривер, память и промпт в единый пайплайн
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever_wrapper,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    return_source_documents=True,
    verbose=False
)

# Интерактивный цикл диалога
print("Продвинутый RAG чат‑бот готов. Введите exit для выхода.")
while True:
    user_input = input("Вы: ")
    if user_input.lower() in ["exit", "quit"]:
        print("До свидания.")
        break
    result = qa_chain.invoke({"question": user_input})
    print("Бот:", result["answer"])
    print()