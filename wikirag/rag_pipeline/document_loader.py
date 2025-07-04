from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader, CSVLoader
from typing import List
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_and_split(file_paths: List[Path], chunk_size=500, chunk_overlap=100) -> List[Document]:
    all_docs = []

    for path in file_paths:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            loader = TextLoader(str(path))
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".md":
            loader = UnstructuredMarkdownLoader(str(path))
        elif suffix == ".csv":
            loader = CSVLoader(str(path))
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(all_docs)
