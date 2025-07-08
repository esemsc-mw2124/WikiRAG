from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from wikirag.rag_pipeline import (   
    create_qa_chain,
    create_conversational_qa_chain,
    load_and_split,
    get_vectorstore,
    DATA_DIR,
    INDEX_DIR,
)

from .router import route           
from .tools import (              
    WikiSingleQATool,
    WikiChatQATool,
    WebSearchTool,
)

def build_chains(article_txt: str | Path, index_name: str):
    chunks = load_and_split([DATA_DIR / article_txt])
    db = get_vectorstore(chunks, INDEX_DIR / index_name)
    single_chain = create_qa_chain(db)
    chat_chain = create_conversational_qa_chain(db)
    return single_chain, chat_chain

def build_tools(single_chain, chat_chain) -> Dict[str, object]:
    return {
        "single": WikiSingleQATool(single_chain),
        "chat":   WikiChatQATool(chat_chain),
        "web":    WebSearchTool(),
    }

def answer_question(
    article_name: str,
    user_input: str,
    chat_history: List[Tuple[str, str]],
    tools: Dict[str, object],
) -> str:
    # Ask the router which branch to take
    decision = route(
        question=user_input,
        chat_history_length=len(chat_history),
    )

    # Run the chosen tool
    if decision == "local_single":
        return tools["single"].run(tool_input=user_input)

    if decision == "local_chat":
        return tools["chat"].run(
            tool_input=user_input,
            chat_history=chat_history,
        )
    # Web search
    reworded = tools["web"].reword_query_for_article(user_input, article_name)
    websearch_results = tools["web"].run(tool_input=reworded)
    return tools["web"].reword_answer_from_websearch(websearch_results, reworded)
