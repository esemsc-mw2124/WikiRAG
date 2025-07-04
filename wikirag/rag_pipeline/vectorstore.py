from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def get_vectorstore(docs, index_path: Path) -> FAISS:
    embeddings = OpenAIEmbeddings()
    if index_path.exists():
        return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(str(index_path))
        return db
