import warnings
warnings.filterwarnings("ignore")
import os
# Отключаем телеметрию и все сетевые запросы к Hugging Face
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

# 1. Инициализация всех моделей и компонентов
# Основная LLM, запущенная локально через LM Studio или Ollama
llm = ChatOpenAI(
    api_key="none",
    base_url="http://localhost:1234/v1/",
    model="qwen/qwen3.5-9b",
    temperature=0.1,
)

# Эмбеддинг-модель загружается из локальной папки, без доступа в интернет
embedding_model_path = "./models/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    model_kwargs={"device": "cpu", "trust_remote_code": False},
    encode_kwargs={"normalize_embeddings": True},
)

# Cross-encoder для точного ранжирования найденных фрагментов
reranker = CrossEncoder(
    model_name="./models/ms-marco-MiniLM-L-6-v2",
    device="cpu",
)

# 2. Загрузка и индексация документов
# Указываем путь к локальному файлу: поддерживаются PDF, DOCX, TXT
source = "test_document.txt"

# Выбираем загрузчик по расширению файла
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

# Разбиваем длинные документы на небольшие фрагменты для поиска
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Создано {len(chunks)} чанков")

# Создаём векторный индекс на основе эмбеддингов и сохраняем его в FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
# Базовый ретривер возвращает 20 лучших кандидатов для последующего реранкинга
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
print("Индекс создан")

# 3. Вспомогательная функция для форматирования документов
def format_docs(docs: List[Document]) -> str:
    # Склеивает содержимое документов через двойной перенос строки
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Техника Query Rewriting — переписывание запроса для улучшения поиска
rewrite_prompt = ChatPromptTemplate.from_template(
    "Перепишите вопрос, чтобы улучшить поиск по документам. "
    "Раскройте аббревиатуры, добавьте синонимы, удалите лишние слова. "
    "Вопрос: {question}\nПереписанный вопрос:"
)
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# 5. Техника HyDE — генерация гипотетического ответа для поиска
hyde_prompt = ChatPromptTemplate.from_template(
    "Сгенерируйте гипотетический ответ на вопрос. Этот ответ будет использован для поиска документов, "
    "поэтому он должен быть информативным и содержать ключевые термины.\nВопрос: {question}\nОтвет:"
)
hyde_chain = hyde_prompt | llm | StrOutputParser()

# 6. Multi-Query — генерация нескольких вариантов запроса
multi_query_prompt = ChatPromptTemplate.from_template(
    "Сгенерируйте 3 различных варианта поискового запроса для вопроса. "
    "Каждый запрос должен быть на отдельной строке.\nВопрос: {question}\nВарианты запросов:"
)
multi_query_chain = multi_query_prompt | llm | StrOutputParser()

def parse_queries(text: str) -> List[str]:
    # Разбивает многострочный ответ на список запросов, удаляя нумерацию
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    queries = []
    for line in lines:
        if line[0].isdigit() and '.' in line[:3]:
            line = line[line.index('.')+1:].strip()
        queries.append(line)
    return queries

# 7. Основная стратегия поиска, объединяющая все методы
def multi_strategy_retrieve(question: str) -> List[Document]:
    # Переписываем исходный запрос
    rewritten = rewrite_chain.invoke({"question": question})
    # Генерируем гипотетический ответ
    hyde_doc = hyde_chain.invoke({"question": question})
    # Получаем несколько вариантов запроса
    queries_text = multi_query_chain.invoke({"question": question})
    queries = parse_queries(queries_text)
    # Собираем все уникальные запросы включая исходный, переписанный и гипотетический
    all_queries = [question, rewritten, hyde_doc] + queries
    all_queries = list(dict.fromkeys(all_queries))

    # Выполняем поиск по каждому запросу и собираем все найденные фрагменты
    all_docs = []
    for q in all_queries:
        docs = base_retriever.invoke(q)
        all_docs.extend(docs)

    # Удаляем дубликаты по содержимому
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    # Применяем кросс-энкодер для точного ранжирования по релевантности исходному вопросу
    if unique_docs:
        pairs = [(question, doc.page_content) for doc in unique_docs]
        scores = reranker.predict(pairs)
        sorted_pairs = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
        # Возвращаем только 4 наиболее релевантных фрагмента
        return [doc for doc, _ in sorted_pairs[:4]]
    return []

# Оборачиваем функцию в Runnable для использования в пайплайне
retrieve_runnable = RunnableLambda(lambda q: multi_strategy_retrieve(q))

# 8. Генерация итогового ответа на основе найденных фрагментов
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Ты - полезный ассистент. Отвечай на вопрос, используя только информацию из предоставленного контекста. Если ответа нет в контексте, скажи: 'Я не знаю, в документах этого нет.'"),
    ("human", "Контекст:\n{context}\n\nВопрос: {question}")
])
def generate_answer(question: str, docs: List[Document]) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

generate_runnable = RunnableLambda(lambda x: generate_answer(x["question"], x["docs"]))

# 9. Оценка качества ответа с помощью LLM-as-a-Judge
judge_prompt = ChatPromptTemplate.from_template(
    "Оцени faithfulness ответа, то есть насколько ответ соответствует контексту. "
    "Ответь одним числом от 0 до 1, где 1 означает полное соответствие, 0 - полную галлюцинацию.\n"
    "Контекст:\n{context}\n\nВопрос: {question}\nОтвет: {answer}\nОценка:"
)
judge_chain = judge_prompt | llm | StrOutputParser()
def evaluate_faithfulness(question: str, answer: str, docs: List[Document]) -> float:
    context = "\n\n".join(doc.page_content for doc in docs)
    try:
        score_text = judge_chain.invoke({"context": context, "question": question, "answer": answer})
        score = float(score_text.strip())
        return max(0.0, min(1.0, score))  # ограничиваем значение отрезком 0–1
    except:
        return 0.5  # значение по умолчанию при ошибке

# 10. Сборка полного пайплайна с использованием LCEL
full_chain = (
    # Первый шаг: передаём вопрос дальше
        RunnablePassthrough()
        # Второй шаг: параллельно получаем вопрос и найденные документы
        | {
            "question": lambda x: x["question"],
            "docs": retrieve_runnable,
        }
        # Третий шаг: генерируем ответ на основе вопроса и документов
        | {
            "question": lambda x: x["question"],
            "docs": lambda x: x["docs"],
            "answer": generate_runnable,
        }
        # Четвёртый шаг: вычисляем оценку faithfulness для полученного ответа
        | {
            "question": lambda x: x["question"],
            "answer": lambda x: x["answer"],
            "docs": lambda x: x["docs"],
            "faithfulness": lambda x: evaluate_faithfulness(x["question"], x["answer"], x["docs"]),
        }
)

# 11. Интерактивный цикл общения с ботом
print("Продвинутый RAG-бот офлайн готов. Введите exit для выхода.")
while True:
    user_input = input("Вы: ")
    if user_input.lower() in ["exit", "quit"]:
        print("До свидания.")
        break
    result = full_chain.invoke({"question": user_input})
    print(f"Бот: {result['answer']}")
    print(f"Оценка faithfulness: {result['faithfulness']:.2f}\n")