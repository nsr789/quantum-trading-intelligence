# src/knowledge/vectorstore.py
"""
Chroma vector-store – compatible with new Chroma client & streamlit-cloud FS.
"""

from __future__ import annotations
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Paths ────────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
STORE_PATH = ROOT / ".cache" / "chroma"
STORE_PATH.mkdir(parents=True, exist_ok=True)

# Single *object* – NOT a function
EMBED = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# ──────────────────────────────────────────────────────────────────────────────
def _seed_docs() -> List[dict]:
    base = [
        ("Apple Inc. designs, manufactures and markets smartphones.", "AAPL"),
        ("Microsoft develops, licenses and supports software products.", "MSFT"),
        ("Alphabet is the parent company of Google, focusing on internet services.", "GOOGL"),
    ]
    return [{"content": txt, "ticker": tic, "id": f"doc-{i}"} for i, (txt, tic) in enumerate(base)]


def _client() -> chromadb.api.ClientAPI:
    """Persistent when possible, in-memory fallback on read-only FS."""
    try:
        return chromadb.PersistentClient(path=str(STORE_PATH), settings=Settings(allow_reset=True))
    except RuntimeError:                      # read-only env (Streamlit Cloud)
        return chromadb.Client(Settings(allow_reset=True))


def load_vectorstore():
    client = _client()

    # If an incompatible collection already exists -> delete & recreate
    if "company_docs" in [c.name for c in client.list_collections()]:
        try:
            col = client.get_collection("company_docs")
            _ = col.count()      # triggers Chroma config validation
        except Exception:
            client.delete_collection("company_docs")

    if "company_docs" not in [c.name for c in client.list_collections()]:
        col = client.create_collection("company_docs", embedding_function=EMBED)
    else:
        col = client.get_collection("company_docs")  # embedding_function already stored

    if col.count() == 0:
        docs = _seed_docs()
        col.add(
            documents=[d["content"] for d in docs],
            metadatas=[{"ticker": d["ticker"]} for d in docs],
            ids=[d["id"] for d in docs],
        )
    return col
