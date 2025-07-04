from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever


def create_qa_chain(vectorstore) -> RetrievalQA:
    # vanilla retriever: retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=ChatOpenAI(model_name="gpt-4o-mini")
    )
    
    prompt = PromptTemplate.from_template("""
    Use the context below to answer the question.
    If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question: {question}
    Answer:""")
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

def create_conversational_qa_chain(vectorstore):
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=ChatOpenAI(model_name="gpt-4o-mini")
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer", 
    )

    custom_prompt = PromptTemplate.from_template(
        """
        You are a helpful assistant answering questions based on the context below.
        Only use the provided context to answer the question.
        If the answer cannot be found in the context, say "The information is not contained in the provided data."
        Do NOT make up answers.

        {context}

        Chat History:
        {chat_history}

        Question: {question}
        Answer:
        """
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
