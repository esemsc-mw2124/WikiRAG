from typing import List, Tuple, Optional, Type, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


class SingleQAInput(BaseModel):
    """Input schema for a stand-alone question."""
    question: str = Field(..., description="A stand-alone question to answer")


class WikiSingleQATool(BaseTool):
    """Answer a single question from the local Wikipedia index."""
    
    name: str = "wikipedia_single_qa"
    description: str = (
        "Answer a one-off question using the stored Wikipedia index. "
        "Use this when there is no prior chat history."
    )
    args_schema: Type[BaseModel] = SingleQAInput
    _qa_chain: Any = PrivateAttr()

    def __init__(self, qa_chain, **kwargs):
        super().__init__(**kwargs)
        self._qa_chain = qa_chain

    def _run(self, question: str) -> str:
        """Pass the user's question through the RetrievalQA chain."""
        return self._qa_chain.invoke({"query": question})["result"]



class ChatQAInput(BaseModel):
    """Input schema for a follow-up Q-A turn."""
    question: str = Field(..., description="The next user question")
    chat_history: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="Earlier (question, answer) pairs in the dialogue",
    )

class WikiChatQATool(BaseTool):
    """Answer follow-up questions, taking chat history into account."""
    
    name: str = "wikipedia_chat_qa"
    description: str = (
        "Answer a follow-up question using the stored Wikipedia index, "
        "taking the previous dialogue context into account."
    )
    args_schema: Type[BaseModel] = ChatQAInput
    _conv_chain: Any = PrivateAttr()

    def __init__(self, conv_chain, **kwargs):
        super().__init__(**kwargs)
        self._conv_chain = conv_chain

    def _run(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """Pass the question + history through the conversational chain."""
        chat_history = chat_history or []
        return self._conv_chain.invoke(
            {"question": question, "chat_history": chat_history}
        )["answer"]
    
class WebSearchInput(BaseModel):
    """Just the query string – nothing else."""
    query: str = Field(..., description="What you want to look up online")


class WebSearchTool(BaseTool):
    """Fetches top-5 DuckDuckGo snippets for fresh information."""

    name: str = "web_search"
    description: str = (
        "Search the web when the Wikipedia page might be outdated. "
        "Returns the first five result snippets."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def reword_query_for_article(user_input: str, article_name: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant that reformulates user queries for web search.",
            ),
            (
                "human",
                "Reword this question so that it explicitly asks about the topic '{article}': {question}"
            ),
        ])
        chain = prompt | llm
        return chain.invoke({"article": article_name, "question": user_input}).content


    def _run(self, query: str) -> str:
        with DDGS() as ddgs:

            results = ddgs.text(query)
            if not results:
                return "No web results found."
            lines = [f"{r['title']} ⇢ {r['body']}" for r in results if 'title' in r and 'body' in r]
            return "\n".join(lines[:5])

    def reword_query_for_article(self, user_input: str, article_name: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant that reformulates user queries for web search.",
            ),
            (
                "human",
                "Reword this question so that it explicitly asks about the topic '{article}': {question}"
            ),
        ])
        chain = prompt | llm
        return chain.invoke({"article": article_name, "question": user_input}).content

    def reword_answer_from_websearch(self, websearch_results: str, question: str) -> str:
        """Uses an LLM to turn raw web search results into a natural-language answer."""
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant. You will be given several brief web search results "
                "and a user question. Your job is to synthesize the search results into a direct, informative answer."
            ),
            (
                "human",
                "Based on the following web search results:\n\n{results}\n\nAnswer the question:\n{question}"
            ),
        ])
        chain = prompt | llm
        return chain.invoke({
            "results": websearch_results,
            "question": question
        }).content