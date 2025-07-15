# src/knowledge/bootstrap.py  (run once, or guard with a joblib cache)
import wikipedia, chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from .vectorstore import _client   # reuse the helper

COMPANIES = {"AAPL": "Apple Inc.", "MSFT": "Microsoft", "GOOGL": "Alphabet Inc."}
EMBED = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def enrich():
    cli = _client()
    col = cli.get_or_create_collection("company_docs", embedding_function=EMBED)
    for tic, title in COMPANIES.items():
        try:
            page = wikipedia.page(title, auto_suggest=False)
            col.add(
                documents=[page.summary],
                metadatas=[{"ticker": tic}],
                ids=[f"{tic}-wiki"],
            )
        except Exception:  # noqa: S110
            pass
