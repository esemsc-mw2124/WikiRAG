# WikiRAG

A simple RAG system that retrieves information from Wikipedia articles.

## 🔧 Setup (with [`uv`](https://github.com/astral-sh/uv))

```bash
# Install uv (if not already done)
curl -Ls https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone https://github.com/esemsc-mw2124/WikiRAG.git
cd WikiRAG

# Install dependencies and create virtual environment
uv sync
```
Then, in the root of the repository create a `.env` file that contains an OpenAI API key:
```
OPENAI_API_KEY=sk-xxxxxxxxxxx[...]
```

