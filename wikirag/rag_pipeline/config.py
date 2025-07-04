from pathlib import Path
from dotenv import load_dotenv

repo_root = Path(__file__).resolve().parents[2]
load_dotenv(repo_root / ".env")

DATA_DIR = repo_root / "wikirag" / "data"
INDEX_DIR = repo_root / "wikirag" / "index"
