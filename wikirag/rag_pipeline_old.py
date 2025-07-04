from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load .env
repo_root = Path(__file__).resolve().parents[1]
load_dotenv(repo_root / ".env")

# Paths
script_dir = Path(__file__).parent
file_path = script_dir / "data" / "millennium_falcon.txt"
index_path = script_dir / "index" / "star_wars_index"

# Load and chunk documents
loader = TextLoader(str(file_path))
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embedder
embeddings = OpenAIEmbeddings()

# Load existing index or build and save it
if index_path.exists():
    db = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
else:
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(index_path))

# Use retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define a prompt template for the QA chain
prompt = PromptTemplate.from_template("""
Use the context below to answer the question.
If the answer is not in the context, say "The Wikipedia page does not contain this information"

Context:
{context}

Question: {question}
Answer:""")

# Initialize the OpenAI chat model (GPT-4) with zero temperature for deterministic output
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Create a RetrievalQA chain using the retriever and custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,  # Return source documents with the answer
    chain_type_kwargs={"prompt": prompt}
)

# Define a sample query
query = "Who is albert einstein?"

# Run the QA chain with the query
result = qa_chain.invoke({"query": query})

# Print the answer
print("Answer:", result["result"])

# Print the source documents used to generate the answer
for doc in result["source_documents"]:
    print("\n--- Source ---")
    print(doc.metadata)
    print(doc.page_content[:300])