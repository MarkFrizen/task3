from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate

# -------------------- 1. Подключение к LLM (локально, без интернета) --------------------
llm = ChatOpenAI(
    api_key="none",                 # для локального сервера ключ не нужен, но формат API требует значение
    base_url="http://192.168.8.11:1234/v1/",  # адрес LM Studio (должен быть доступен в локальной сети)
    model="qwen/qwen3.5-9b",        # имя модели, загруженной в LM Studio
    temperature=0.1,                # небольшая случайность в ответах для естественности
)

# -------------------- 2. Загрузка документа (только локальные файлы) --------------------
source = "test_document.txt"  # поддерживаются только локальные файлы: .pdf, .docx, .txt
if source.endswith('.pdf'):
    loader = PyPDFLoader(source)
elif source.endswith('.docx'):
    loader = Docx2txtLoader(source)
elif source.endswith('.txt'):
    loader = TextLoader(source, encoding='utf-8')
else:
    # URL не поддерживаются: это нарушило бы условие «полностью офлайн»
    raise ValueError("Поддерживаются только локальные файлы: PDF, DOCX, TXT")
documents = loader.load()
print(f"Загружено {len(documents)} страниц")

# -------------------- 3. Умная нарезка текста с перекрытием --------------------
# Длинный документ разбивается на небольшие чанки, чтобы они помещались в контекст модели.
# Перекрытие (chunk_overlap) нужно, чтобы не потерять смысл на границах фрагментов.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,           # целевой размер чанка (в символах)
    chunk_overlap=50,        # перекрытие между соседними чанками
    length_function=len,      # функция подсчёта длины (по количеству символов)
    separators=["\n\n", "\n", " ", ""]  # приоритеты разбиения: сначала по двойным переносам, потом по одиночным и т.д.
)
chunks = text_splitter.split_documents(documents)
print(f"Создано {len(chunks)} чанков")

# -------------------- 4. Векторный индекс (полностью офлайн) --------------------
# УКАЖИ ПУТЬ К ЛОКАЛЬНОЙ МОДЕЛИ ЭМБЕДДИНГОВ, которую ты скачал заранее.
# Без этого при первом запуске будет попытка скачать модель из интернета.
local_embedding_path = "/home/ubuntu/models/paraphrase-multilingual-MiniLM-L12-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=local_embedding_path,      # путь к папке с файлами модели (не URL)
    model_kwargs={"trust_remote_code": True},  # требуется для корректной загрузки некоторых моделей
    encode_kwargs={"device": "cpu"}      # "cpu" — для работы без GPU; если есть GPU, поставь "cuda"
)

# Создаём векторное хранилище на основе чанков и их эмбеддингов.
vectorstore = FAISS.from_documents(chunks, embeddings)
# Настраиваем ретривер: при запросе он будет возвращать 4 наиболее релевантных чанка.
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("Индекс создан")

# -------------------- 5. История диалога (память) --------------------
# Позволяет боту «помнить» предыдущие вопросы и ответы в рамках одной сессии.
memory = ConversationBufferMemory(
    memory_key="chat_history",   # ключ, под которым история будет передаваться в промпт
    return_messages=True,        # возвращает историю в виде списка сообщений
    output_key="answer"          # ключ, куда запишется ответ модели в итоговом словаре
)

# -------------------- 6. Промпт: правила поведения и формат ввода --------------------
prompt_template = ChatPromptTemplate.from_messages([
    # Системная инструкция: ограничивает поведение модели и задаёт стиль ответов.
    ("system", "Ты - полезный ассистент. Отвечай на вопрос, используя только информацию из предоставленного контекста. Если ответа нет в контексте, скажи: 'Я не знаю, в документах этого нет'."),
    # Шаблон для пользовательского запроса: сначала контекст, потом вопрос.
    ("human", "Контекст:\n{context}\n\nВопрос: {question}")
])

# -------------------- 7. Сборка RAG-цепочки --------------------
# Объединяет LLM, ретривер, память и промпт в единый пайплайн.
# При каждом вопросе:
#   1) из памяти берётся история диалога;
#   2) ретривер ищет релевантные чанки в векторном индексе;
#   3) формируется промпт (системное сообщение + контекст + вопрос);
#   4) LLM генерирует ответ.
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    return_source_documents=True,  # возвращает найденные документы (полезно для проверки источников)
    verbose=False                  # не выводит отладочные логи в консоль
)

# -------------------- 8. Интерактивный цикл диалога --------------------
print("Чат-бот готов. Введите 'exit' для выхода.")
while True:
    user_input = input("Вы: ")
    if user_input.lower() in ["exit", "quit"]:
        print("До свидания.")
        break
    result = qa_chain.invoke({"question": user_input})
    print("Бот:", result["answer"])
    print()