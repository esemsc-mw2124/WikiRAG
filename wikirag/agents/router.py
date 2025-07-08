from typing import Literal
from langchain_openai import ChatOpenAI

_ROUTER_SYSTEM_PROMPT = """\
You are a routing assistant in a Retrieval-Augmented Generation system.

Decide which action can best answer the user's question.
Respond with ONLY one of these tokens (no quotes, no extra text):

    local_single   - The question is stand-alone AND can be answered from a static Wikipedia page.
    local_chat     - The user is in the middle of a conversation; use chat history AND the static page.
    search         - Information may have changed SINCE the last Wikipedia edit, or is not covered there. A fresh web lookup is needed.

Rules & hints
-------------
• Historical facts don't ususally change → local_*
• If the question contains words like "latest", "current", "this year", "upcoming", "new album", "CEO now", → search
"""

def route(
    question: str,
    chat_history_length: int = 0,
    llm: ChatOpenAI | None = None,
) -> Literal["local_single", "local_chat", "search"]:
    
    llm = llm or ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Compose the dynamic user prompt.
    user_prompt = (
        f"Chat history length: {chat_history_length}\n\n"
        f"Question: {question}"
    )

    decision = llm.invoke(
        [_ROUTER_SYSTEM_PROMPT, user_prompt]
    ).content.strip().lower()

    # Sanity-check; default to search on invalid output.
    if decision not in {"local_single", "local_chat", "search"}:
        decision = "search"

    return decision