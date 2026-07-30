from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. Подключение к LM Studio
llm = ChatOpenAI(
    api_key="none",
    base_url="http://192.168.8.11:1234/v1/",
    model="qwen/qwen3.5-9b",
    temperature=0.1,
)

# 2. Загрузка документа
source = "bau_fahj.pdf"  # замените на свой файл

if source.endswith('.pdf'):      # .endswith() – правильно
    loader = PyPDFLoader(source)
elif source.endswith('.docx'):
    loader = Docx2txtLoader(source)
elif source.startswith('http'):
    loader = WebBaseLoader(source)
else:
    raise ValueError("Поддерживаются: PDF, DOCX, веб-ссылка")

documents = loader.load()
print(f"Загружено {len(documents)} страниц")

# 3. Нарезка
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Создано {len(chunks)} чанков")

# 4. Векторный индекс
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("Индекс создан")

# 5. История диалога (ручное управление)
chat_history = []

# 6. Промпт для ретривера с историей
history_aware_prompt = ChatPromptTemplate.from_messages([
    ("system", "Учитывая историю диалога, сформулируй поисковый запрос для получения релевантных документов."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=history_aware_prompt
)

# 7. Промпт для ответа (System + Контекст + Вопрос)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты - полезный ассистент. Отвечай на вопрос, используя только информацию из предоставленного контекста. Если ответа нет в контексте, скажи: 'Я не знаю, в документах этого нет'."),
    ("system", "Контекст:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)

# 8. Итоговая RAG-цепочка
rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=combine_docs_chain
)

# 9. Цикл диалога
print("Чат-бот на LM Studio готов. Задавайте вопросы по документу. Введите 'exit' для выхода.")

while True:
    user_input = input("Вы: ")
    if user_input.lower() in ["exit", "quit"]:
        print("До свидания.")
        break

    result = rag_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    answer = result["answer"]
    print("Бот:", answer)
    print()

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=answer))