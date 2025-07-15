# ── src/knowledge/vectorstore.py ─────────────────────────────────────────────
"""
Persistent Chroma collection (no legacy config).  
Uses MiniLM embeddings via Chroma’s helper wrapper – complies with the new
embedding-function contract (name(), version()).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --------------------------------------------------------------------------- #
ROOT        = Path(__file__).resolve().parents[2]
STORE_PATH  = ROOT / ".cache" / "chroma"
STORE_PATH.mkdir(parents=True, exist_ok=True)

EMBED = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def _seed_docs() -> List[dict]:
    examples = [
        ("Apple Inc. designs, manufactures and markets smartphones.", "AAPL"),
        ("Microsoft develops, licenses and supports software products.", "MSFT"),
        ("Alphabet is the parent company of Google, focusing on internet services.", "GOOGL"),
    ]
    return [{"content": txt, "ticker": tck, "id": f"doc-{i}"} for i, (txt, tck) in enumerate(examples)]


# --------------------------------------------------------------------------- #
def load_vectorstore():
    """Return a Chroma collection, seeding a tiny corpus on first run."""
    try:
        client = chromadb.PersistentClient(
            path=str(STORE_PATH),
            settings=Settings(allow_reset=True),
        )
    except RuntimeError:
        # read-only FS fallback (Streamlit Community Cloud)
        client = chromadb.Client(Settings())

    col = client.get_or_create_collection(
        name="company_docs",
        embedding_function=EMBED,
    )

    if col.count() == 0:
        docs = _seed_docs()
        col.add(
            documents=[d["content"] for d in docs],
            metadatas=[{"ticker": d["ticker"]} for d in docs],
            ids=[d["id"] for d in docs],
        )
    return col
