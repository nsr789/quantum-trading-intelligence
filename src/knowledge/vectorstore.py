# ── src/knowledge/vectorstore.py ─────────────────────────────────────────────
"""
Persistent Chroma vector-store (new client API, no legacy config clashes).

* Uses SentenceTransformerEmbeddingFunction → object has name()/version()
* If a legacy collection exists with a conflicting embedding config,
  it is deleted and recreated automatically – avoids AttributeError.
* Falls back to an in-memory client when the filesystem is read-only
  (Streamlit Community Cloud edge-case).
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
    return [
        {"content": txt, "ticker": tck, "id": f"doc-{i}"}
        for i, (txt, tck) in enumerate(examples)
    ]


def _get_client() -> chromadb.api.ClientAPI:
    """Writable persistent client, else in-memory fallback."""
    try:
        return chromadb.PersistentClient(
            path=str(STORE_PATH),
            settings=Settings(allow_reset=True),
        )
    except RuntimeError:
        # Read-only FS (Streamlit Cloud) → use in-memory client
        return chromadb.Client(Settings(allow_reset=True))


# --------------------------------------------------------------------------- #
def load_vectorstore():
    client = _get_client()

    # --------------------------------------------------------------------- #
    # 1️⃣  Ensure a clean collection with the *correct* embedding function
    # --------------------------------------------------------------------- #
    try:
        col = client.get_collection("company_docs")
        # If an old collection was created with the default embedding config,
        # its persisted config won't match our EMBED object → ValueError.
        # Access a property to trigger validation immediately:
        _ = col.count()
    except Exception:
        # Either collection doesn’t exist OR embedding config mismatched.
        if "company_docs" in [c.name for c in client.list_collections()]:
            client.delete_collection("company_docs")
        col = client.create_collection(
            name="company_docs",
            embedding_function=EMBED,
        )

    # --------------------------------------------------------------------- #
    # 2️⃣  Seed tiny demo corpus once
    # --------------------------------------------------------------------- #
    if col.count() == 0:
        docs = _seed_docs()
        col.add(
            documents=[d["content"] for d in docs],
            metadatas=[{"ticker": d["ticker"]} for d in docs],
            ids=[d["id"] for d in docs],
        )
    return col
