from wikirag.utils import save_article_to_txt
from wikirag.rag_pipeline import DATA_DIR
from wikirag.agents.runner import build_chains, build_tools, answer_question


def main():
    article = input("Enter the Wikipedia article to look up: ")
    file_name = save_article_to_txt(DATA_DIR, article)
    index_name = file_name[:-4]+ "_index"

    single_chain, chat_chain = build_chains(file_name, index_name)
    tools = build_tools(single_chain, chat_chain)
    chat_history = []

    while True:
        user = input("You: ")
        if user.lower() in {"exit", "quit"}:
            break

        answer = answer_question(
            article,
            user,
            chat_history,
            tools=tools,
        )
        print("Bot:", answer, "\n")
        chat_history.append((user, answer))

if __name__ == "__main__":
    main()