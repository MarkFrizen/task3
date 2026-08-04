import warnings
warnings.filterwarnings("ignore")
import os
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.prompts import ChatPromptTemplate

# Подключение локальной LLM
llm = ChatOpenAI(
    api_key="none",
    base_url="http://localhost:1234/v1/",
    model="qwen/qwen3.5-9b",
    temperature=0.1,
)

# Загрузка документа
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

# Разбиение текста на чанки
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Создано {len(chunks)} чанков")

# Векторный индекс — эмбеддинги + FAISS
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"device": "cpu"}
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("Индекс создан")

# Промпт
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Ты - полезный ассистент. Отвечай на вопрос, используя только информацию из предоставленного контекста. Если ответа нет в контексте, скажи: Я не знаю, в документах этого нет."),
    ("human", "Контекст:\n{context}\n\nВопрос: {question}")
])

# RAG-цепочка (память встроена в новых версиях langchain)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    return_source_documents=True,
    verbose=False
)

# Интерактивный цикл (история хранится через chat_history)
print("Чат-бот готов. Введите exit для выхода.")
chat_history = []
while True:
    user_input = input("Вы: ")
    if user_input.lower() in ["exit", "quit"]:
        print("До свидания.")
        break
    result = qa_chain.invoke({"question": user_input, "chat_history": chat_history})
    print("Бот:", result["answer"])
    chat_history.append(f"Human: {user_input}")
    chat_history.append(f"AI: {result['answer']}")
    print()