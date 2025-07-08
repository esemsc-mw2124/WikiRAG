import streamlit as st
from wikirag.utils import save_article_to_txt
from wikirag.agents.runner import build_chains, build_tools, answer_question
from wikirag.rag_pipeline import DATA_DIR

import sys
from pathlib import Path


print("cwd:", Path.cwd())
print("sys.executable:", sys.executable)

sys.path.append(str(Path(__file__).resolve().parents[1]))
print("sys.path[0]:", sys.path[0])

try:
    import wikirag
    print("✅ Successfully imported wikirag")
except ModuleNotFoundError as e:
    print("❌", e)


# Add the root directory of your project to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(page_title="WikiRAG Chatbot", layout="wide")

# Session state for memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "tools" not in st.session_state:
    st.session_state.tools = None

if "article_name" not in st.session_state:
    st.session_state.article_name = ""

st.title("🧠 WikiRAG Chatbot")

# Input for Wikipedia article
article = st.text_input("🔍 Wikipedia article:", value=st.session_state.article_name)

if article and st.button("Load Article"):
    file_name = save_article_to_txt(DATA_DIR, article)
    index_name = file_name[:-4] + "_index"

    single_chain, chat_chain = build_chains(file_name, index_name)
    st.session_state.tools = build_tools(single_chain, chat_chain)
    st.session_state.article_name = article
    st.session_state.chat_history = []
    st.success(f"Loaded Wikipedia index for: {article}")

# Chat input
if st.session_state.tools:
    user_input = st.chat_input("Ask something about the article...")

    if user_input:
        answer = answer_question(
            user_input=user_input,
            chat_history=st.session_state.chat_history,
            tools=st.session_state.tools,
            article_name=st.session_state.article_name
        )

        st.session_state.chat_history.append((user_input, answer))

# Display history
for i, (q, a) in enumerate(st.session_state.chat_history):
    st.chat_message("user", avatar="🧑").write(q)
    st.chat_message("assistant", avatar="🤖").write(a)
