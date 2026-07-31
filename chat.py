from langchain_openai import ChatOpenAI                     # для подключения к LM Studio через OpenAI-совместимый API
from langchain_community.document_loaders import (         # загрузчики документов разных форматов
    PyPDFLoader, Docx2txtLoader, WebBaseLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # умная нарезка текста с перекрытием
from langchain_community.embeddings import HuggingFaceEmbeddings      # создание векторных представлений (эмбеддингов)
from langchain_community.vectorstores import FAISS                   # векторная база данных (индекс)
from langchain.memory import ConversationBufferMemory               # хранилище истории диалога (Chat Memory)
from langchain.chains import ConversationalRetrievalChain           # готовая цепочка "поиск + генерация"
from langchain.prompts import ChatPromptTemplate                    # шаблоны для формирования промптов

# -------------------- 1. Подключение к LLM (LM Studio) --------------------
# Используем локальный сервер LM Studio с моделью Qwen 9B
llm = ChatOpenAI(
    api_key="none",                                      # LM Studio не требует ключа
    base_url="http://192.168.8.11:1234/v1/",            # адрес вашего сервера (порт 1234)
    model="qwen/qwen3.5-9b",                             # имя модели, загруженной в LM Studio
    temperature=0.1,                                     # низкая температура для более детерминированных ответов
)

# -------------------- 2. Загрузка документа --------------------
# Поддерживаются: PDF, DOCX, TXT, веб-страницы (URL)
source = "test_document.txt"  # измените на свой файл или URL
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
documents = loader.load()                     # загружаем содержимое документа
print(f"Загружено {len(documents)} страниц")   # для PDF — страницы, для текста — один документ

# -------------------- 3. Умная нарезка с перекрытием --------------------
# Разбиваем длинный текст на куски, чтобы поместить в контекст LLM.
# Перекрытие (overlap) помогает не потерять смысл на границах чанков.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,                # размер одного чанка в символах
    chunk_overlap=50,              # перекрытие (10% от размера)
    length_function=len,           # функция подсчёта длины (символы)
    separators=["\n\n", "\n", " ", ""]  # сначала делим по абзацам, потом по предложениям, словам
)
chunks = text_splitter.split_documents(documents)  # выполняем нарезку
print(f"Создано {len(chunks)} чанков")

# -------------------- 4. Создание векторного индекса (FAISS) --------------------
# Превращаем текст чанков в числа (эмбеддинги), чтобы находить похожие по смыслу.
# Используем мультиязычную модель, которая хорошо работает с русским языком.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)   # строим индекс
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # при поиске выдаём 4 лучших чанка
print("Индекс создан")

# -------------------- 5. История диалога (Chat Memory) --------------------
# Хранит все предыдущие вопросы и ответы, чтобы бот помнил контекст беседы.
memory = ConversationBufferMemory(
    memory_key="chat_history",   # ключ, по которому история будет доступна в цепочке
    return_messages=True,        # возвращать сообщения в виде объектов (не строк)
    output_key="answer"          # откуда брать ответ для сохранения в память
)

# -------------------- 6. Промпт: System + Контекст + Вопрос --------------------
# Задаём поведение модели: отвечать строго по контексту, если не знает — признаваться.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Ты - полезный ассистент. Отвечай на вопрос, используя только информацию из предоставленного контекста. Если ответа нет в контексте, скажи: 'Я не знаю, в документах этого нет'."),
    ("human", "Контекст:\n{context}\n\nВопрос: {question}")
])

# -------------------- 7. Сборка RAG-цепочки --------------------
# Объединяем LLM, ретривер, память и промпт в единый пайплайн.
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},  # подставляем наш промпт
    return_source_documents=True,  # чтобы можно было посмотреть, откуда взят ответ (для отладки)
    verbose=False                  # отключить лишний вывод
)

# -------------------- 8. Интерактивный цикл диалога --------------------
print("Чат-бот готов. Введите 'exit' для выхода.")
while True:
    user_input = input("Вы: ")
    if user_input.lower() in ["exit", "quit"]:
        print("До свидания.")
        break
    result = qa_chain.invoke({"question": user_input})  # вызываем цепочку с вопросом
    print("Бот:", result["answer"])                    # выводим ответ
    print()