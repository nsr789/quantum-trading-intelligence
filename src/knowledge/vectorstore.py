from __future__ import annotations
import os
os.environ.setdefault("CHROMA_DB_IMPL", "duckdb+parquet")  # skip SQLite

import pathlib
from typing import List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

_ROOT = pathlib.Path(__file__).resolve().parents[2]
STORE_PATH = _ROOT / ".cache" / "chroma"
EMBED = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def _default_docs() -> List[dict]:
    txt = [
        ("Apple Inc. designs, manufactures and markets smartphones.", "AAPL"),
        ("Microsoft develops, licenses and supports software products.", "MSFT"),
        ("Alphabet is the parent company of Google, focusing on internet services.", "GOOGL"),
    ]
    return [{"content": c, "ticker": t, "id": f"doc-{i}"} for i, (c, t) in enumerate(txt)]


def load_vectorstore():
    """Return a Chroma collection, creating it with default docs if empty."""
    client = chromadb.PersistentClient(path=str(STORE_PATH))
    col = client.get_or_create_collection("company_docs", embedding_function=EMBED)

    if col.count() == 0:  # seed with tiny corpus
        docs = _default_docs()
        col.add(
            documents=[d["content"] for d in docs],
            metadatas=[{"ticker": d["ticker"]} for d in docs],
            ids=[d["id"] for d in docs],
        )
    return col
