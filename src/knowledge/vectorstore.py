# ── src/knowledge/vectorstore.py ──────────────────────────────────────────────
"""
Light-weight Chroma knowledge-base that ships with a micro-corpus.

Updates vs. the legacy code
───────────────────────────
1.   Uses **chromadb.PersistentClient(path=…, settings=Settings(allow_reset=True))**
     which avoids every legacy configuration key that triggered the
     `ValueError: You are using a deprecated configuration of Chroma`.
2.   Embedding function is passed as a **callable** instead of the old
     `SentenceTransformerEmbeddingFunction` helper (works identically).
3.   Falls back to an *in-memory* client when the filesystem is read-only
     (Streamlit Cloud edge-case).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb
from chromadb.api.client import ClientAPI
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Persistent storage folder ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # project root
STORE_PATH = ROOT / ".cache" / "chroma"
STORE_PATH.mkdir(parents=True, exist_ok=True)       # ok if already exists

# ── Embedding function (small & fast) ────────────────────────────────────────
_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def _embed(texts: List[str]) -> List[List[float]]:
    """Return dense embeddings – signature expected by Chroma."""
    return _MODEL.encode(texts, normalize_embeddings=True).tolist()

# ── Tiny default corpus ──────────────────────────────────────────────────────
def _default_docs() -> List[dict]:
    data = [
        ("Apple Inc. designs, manufactures and markets smartphones.", "AAPL"),
        ("Microsoft develops, licenses and supports software products.", "MSFT"),
        ("Alphabet is the parent company of Google, focusing on internet services.", "GOOGL"),
    ]
    return [{"content": c, "ticker": t, "id": f"doc-{i}"} for i, (c, t) in enumerate(data)]

# ── Public helper ────────────────────────────────────────────────────────────
def load_vectorstore() -> ClientAPI.Collection:
    """
    Return a Chroma collection, creating and **seeding** it if empty.

    • Works with Chroma ≥ 0.5 (no legacy keys → no ValueError)
    • Automatically falls back to an in-memory client if the home
      directory is not writable (rare on some managed hosts).
    """
    try:
        client = chromadb.PersistentClient(
            path=str(STORE_PATH),
            settings=Settings(allow_reset=True),  # modern setting
        )
    except RuntimeError:
        # e.g. read-only file-system on Streamlit Cloud
        client = chromadb.Client(Settings())      # purely in-memory

    col = client.get_or_create_collection(
        name="company_docs",
        embedding_function=_embed,               # pass callable
    )

    # ── first-run seeding ───────────────────────────────────────────────────
    if col.count() == 0:
        docs = _default_docs()
        col.add(
            documents=[d["content"] for d in docs],
            metadatas=[{"ticker": d["ticker"]} for d in docs],
            ids=[d["id"] for d in docs],
        )

    return col
