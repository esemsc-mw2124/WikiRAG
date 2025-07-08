from rag_pipeline.config import DATA_DIR, INDEX_DIR
from rag_pipeline.document_loader import load_and_split
from rag_pipeline.vectorstore import get_vectorstore
from rag_pipeline.qa_engine import create_qa_chain, create_conversational_qa_chain
from wikirag.utils.wikipedia_parser import save_article_to_txt

from typing import List, Union

def setup_qa(file_names: Union[str, List[str]], index_name: str, chatbot_mode: bool = True):
    # Normalize input to a list
    if isinstance(file_names, str):
        file_paths = [DATA_DIR / file_names]
    else:
        file_paths = [DATA_DIR / name for name in file_names]

    index_path = INDEX_DIR / index_name

    chunks = load_and_split(file_paths)
    db = get_vectorstore(chunks, index_path)
    qa_chain = create_conversational_qa_chain(db) if chatbot_mode else create_qa_chain(db)
    return qa_chain

def run_query(qa_chain, query: str, verbose_sources: bool = False):
    result = qa_chain.invoke({"query": query})
    print("Answer:", result["result"])
    if verbose_sources:
        for doc in result["source_documents"]:
            print("\n--- Source ---")
            print(doc.metadata)
            print(doc.page_content[:300])

def run_chatbot():
    print("\nAsk anything about the Wikipedia page. Type 'exit' to quit.\n")
    chat_history = []

    while True:
        question = input("You: ")
        if question.lower() in {"exit", "quit"}:
            break
        response = qa_chain.invoke({"question": question, "chat_history": chat_history})
        print(f"WikiBot: {response['answer']}\n")
        chat_history.append((question, response["answer"]))


if __name__ == "__main__":
    article = input("Enter the Wikipedia article to look up: ")
    file_name = save_article_to_txt(DATA_DIR, article)
    index_name = file_name[:-4]+ "_index"
    qa_chain = setup_qa(file_name, index_name)

    run_chatbot()
