from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate

# -------------------- 1. Подключение к LLM локально --------------------
llm = ChatOpenAI(
    api_key="none",
    base_url="http://192.168.8.11:1234/v1/",  # адрес LM Studio
    model="qwen/qwen3.5-9b",
    temperature=0.1,
)

# -------------------- 2. Загрузка документа только локальные файлы --------------------
source = "test_document.txt"
if source.endswith('.pdf'):
    loader = PyPDFLoader(source)
elif source.endswith('.docx'):
    loader = Docx2txtLoader(source)
elif source.endswith('.txt'):
    loader = TextLoader(source, encoding='utf-8')
else:
    raise ValueError("Поддерживаются только локальные файлы: PDF, DOCX, TXT")
documents = loader.load()
print(f"Загружено {len(documents)} страниц")

# -------------------- 3. Умная нарезка текста с перекрытием --------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Создано {len(chunks)} чанков")

# -------------------- 4. Векторный индекс офлайн --------------------
# ВАРИАНТ 1: если у тебя уже есть модель в кэше сеть не потребуется
# Раскомментируй одну из этих строк и закомментируй local_embedding_path ниже:
# embedding_model_name = "all-MiniLM-L6-v2"
# embedding_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
# ВАРИАНТ 2: если используешь свою локальную папку — оставь путь, но убедись, что она реально существует
local_embedding_path = "/home/ubuntu/models/paraphrase-multilingual-MiniLM-L12-v2"
embedding_model_name = local_embedding_path
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"device": "cpu"}  # поставь cuda, если есть GPU
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("Индекс создан")

# -------------------- 5. История диалога память --------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# -------------------- 6. Промпт --------------------
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Ты - полезный ассистент. Отвечай на вопрос, используя только информацию из предоставленного контекста. Если ответа нет в контексте, скажи: Я не знаю, в документах этого нет."),
    ("human", "Контекст:\n{context}\n\nВопрос: {question}")
])

# -------------------- 7. Сборка RAG-цепочки --------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    return_source_documents=True,
    verbose=False
)

# -------------------- 8. Интерактивный цикл диалога --------------------
print("Чат-бот готов. Введите exit для выхода.")
while True:
    user_input = input("Вы: ")
    if user_input.lower() in ["exit", "quit"]:
        print("До свидания.")
        break
    result = qa_chain.invoke({"question": user_input})
    print("Бот:", result["answer"])
    print()