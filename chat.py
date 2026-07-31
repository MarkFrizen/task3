from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate

# -------------------- 1. Подключение к LLM --------------------
# Используем локальный сервер LM Studio с моделью Qwen 9B
llm = ChatOpenAI(
    api_key="none",
    base_url="http://192.168.8.11:1234/v1/",
    model="qwen/qwen3.5-9b",
    temperature=0.1,
)

# -------------------- 2. Загрузка документа --------------------
# Поддерживаются: PDF, DOCX, TXT, веб-страницы
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

# -------------------- 3. Умная нарезка с перекрытием --------------------
# Разбиваем длинный текст на куски, чтобы поместить в контекст LLM.
# Перекрытие помогает не потерять смысл на границах чанков.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Создано {len(chunks)} чанков")

# -------------------- 4. Создание векторного индекса --------------------
# Превращаем текст чанков в числа, чтобы находить похожие по смыслу.
# Используем мультиязычную модель, которая хорошо работает с русским языком.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("Индекс создан")

# -------------------- 5. История диалога --------------------
# Хранит все предыдущие вопросы и ответы, чтобы бот помнил контекст беседы.
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
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
    combine_docs_chain_kwargs={"prompt": prompt_template},
    return_source_documents=True,
    verbose=False
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