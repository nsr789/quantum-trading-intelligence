# ── src/knowledge/vectorstore.py ─────────────────────────────────────────────
"""
Persistent Chroma vector-store compatible with the *new* Chroma client
(avoids legacy-config & embedding-function conflicts).
"""

from __future__ import annotations
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --- locations & embedding function ----------------------------------------
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
    return [
        {"content": txt, "ticker": tck, "id": f"doc-{i}"}
        for i, (txt, tck) in enumerate(examples)
    ]


# ---- client helpers --------------------------------------------------------
def _get_client() -> chromadb.api.ClientAPI:
    """Writable persistent client; falls back to in-memory on read-only FS."""
    try:
        return chromadb.PersistentClient(
            path=str(STORE_PATH),
            settings=Settings(allow_reset=True),
        )
    except RuntimeError:
        # Streamlit Community Cloud uses a read-only filesystem
        return chromadb.Client(Settings(allow_reset=True))


# ---- public loader ---------------------------------------------------------
def load_vectorstore():
    client = _get_client()

    # 1. ensure collection has the correct embedding-function config
    try:
        col = client.get_collection("company_docs")
        _ = col.count()            # triggers config validation
    except Exception:
        if "company_docs" in [c.name for c in client.list_collections()]:
            client.delete_collection("company_docs")
        col = client.create_collection(
            name="company_docs",
            embedding_function=EMBED,
        )

    # 2. seed a tiny demo corpus once
    if col.count() == 0:
        docs = _seed_docs()
        col.add(
            documents=[d["content"] for d in docs],
            metadatas=[{"ticker": d["ticker"]} for d in docs],
            ids=[d["id"] for d in docs],
        )
    return col
