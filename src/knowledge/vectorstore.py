# ── src/knowledge/vectorstore.py ───────────────────────────────────────────
"""
Modern Chroma client (0.5+) -- no legacy config keys ⇒ no ValueError.
"""

from __future__ import annotations
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

ROOT        = Path(__file__).resolve().parents[2]
STORE_PATH  = ROOT / ".cache" / "chroma"
STORE_PATH.mkdir(parents=True, exist_ok=True)

_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def _embed(texts: List[str]) -> List[List[float]]:
    return _MODEL.encode(texts, normalize_embeddings=True).tolist()

_SEED = [
    ("Apple Inc. designs, manufactures and markets smartphones.", "AAPL"),
    ("Microsoft develops, licenses and supports software products.", "MSFT"),
    ("Alphabet is the parent company of Google, focusing on internet services.", "GOOGL"),
]

def _default_docs():
    return [{"content": c, "ticker": t, "id": f"doc-{i}"} for i, (c, t) in enumerate(_SEED)]

# ––––– public helper –––––
def load_vectorstore():
    try:
        client = chromadb.PersistentClient(
            path=str(STORE_PATH),
            settings=Settings(allow_reset=True),   # <- modern key
        )
    except RuntimeError:
        # read-only FS fallback (Streamlit edge-case)
        client = chromadb.Client(Settings())

    col = client.get_or_create_collection(
        "company_docs",
        embedding_function=_embed
    )

    if col.count() == 0:
        docs = _default_docs()
        col.add(
            documents=[d["content"] for d in docs],
            metadatas=[{"ticker": d["ticker"]} for d in docs],
            ids=[d["id"] for d in docs],
        )
    return col
