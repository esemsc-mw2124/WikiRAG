import wikipedia
import re
from pathlib import Path

def get_wikipedia_article(article_name: str, lang: str = "en", interactive: bool = False) -> dict | None:
    """
    Fetches the summary and full page content of a Wikipedia article.
    Args:
        article_name (str): The title or search term for the Wikipedia article.
        lang (str): Language code (default is 'en').
        interactive (bool): If True, shows top 5 search results and lets user pick.
    Returns:
        dict: Contains 'title', 'url', 'summary', and 'content' if found, else None.
    """
    wikipedia.set_lang(lang)
    try:
        if interactive:
            results = wikipedia.search(article_name, results=5)
            if not results:
                print(f"No articles found for '{article_name}'.")
                return None
            print("Select an article:")
            for idx, title in enumerate(results, 1):
                print(f"{idx}. {title}")
            while True:
                try:
                    choice = int(input("Enter the number of the article: "))
                    if 1 <= choice <= len(results):
                        article_name = results[choice - 1]
                        break
                    else:
                        print("Invalid selection. Try again.")
                except ValueError:
                    print("Please enter a valid number.")
        summary = wikipedia.summary(article_name)
        page = wikipedia.page(article_name)
        return {
            "title": page.title,
            "url": page.url,
            "summary": summary,
            "content": page.content
        }
    except wikipedia.exceptions.PageError:
        print(f"Article '{article_name}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None


def save_article_to_txt(file_path: Path, article_name: str, lang: str = "en", interactive: bool = False) -> str:
    """
    Saves the content of a Wikipedia article to a text file at the specified file_path.
    Args:
        file_path (Path): The directory where the file will be saved.
        article_name (str): The article to fetch and save.
    Raises:
        ValueError or RuntimeError from get_wikipedia_article
    """
    article = get_wikipedia_article(article_name, lang=lang, interactive=interactive)
    
    # Clean title: lowercase, replace spaces with underscores, remove special chars
    filename_safe = re.sub(r"[^\w\s-]", "", article['title']).strip().lower().replace(" ", "_") + ".txt"
    full_path = file_path / filename_safe

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(article['content'])

    return str(filename_safe)


