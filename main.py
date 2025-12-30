import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

load_dotenv()

# ---------------- CONFIG ----------------
DATA_PATH = "data/"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------- LOAD PDF ----------------
def load_documents(path: str):
    if not os.path.exists(path):
        return []
    loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# ---------------- VECTOR STORE ----------------
def get_vectorstore(chunks):
    if not chunks:
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    return FAISS.from_documents(chunks, embeddings)

# ---------------- LLM ----------------
def get_llm():
    return ChatGroq(model=LLM_MODEL, temperature=0, groq_api_key=GROQ_API_KEY)

# ---------------- CHAIN ----------------
def get_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful university helpdesk assistant. Use the context to answer."),
        ("human", "{input}")
    ])
    qa_chain = prompt | llm

    store = {}
    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
