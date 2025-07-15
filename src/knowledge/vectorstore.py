# ────────────────────────── src/knowledge/vectorstore.py ─────────────────────
"""
Modern Chroma vector-store helper.

• Works with Chroma ≥0.5 (new client / config validation)
• Automatically recreates the collection if legacy config is detected
• Silently falls back to in-memory DB on read-only hosts (Streamlit Cloud)
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ── Globals -------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
STORE_PATH  = ROOT / ".cache" / "chroma"
STORE_PATH.mkdir(parents=True, exist_ok=True)

EMBED = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def _seed_docs() -> List[dict]:
    corpus = [
        ("Apple Inc. designs, manufactures and markets smartphones.", "AAPL"),
        ("Microsoft develops, licenses and supports software products.", "MSFT"),
        ("Alphabet (Google) focuses on internet services and advertising.", "GOOGL"),
    ]
    return [{"content": txt, "ticker": tic, "id": f"doc-{i}"} for i, (txt, tic) in enumerate(corpus)]


# ── Client helper -------------------------------------------------------------
def _client() -> chromadb.api.ClientAPI:
    """Writable persistent client when possible, otherwise in-memory."""
    try:
        return chromadb.PersistentClient(
            path=str(STORE_PATH),
            settings=Settings(allow_reset=True)  # allow programmatic delete
        )
    except RuntimeError:  # read-only filesystem
        return chromadb.Client(Settings(allow_reset=True))


# ── Public factory ------------------------------------------------------------
def load_vectorstore():
    client = _client()

    # (1) ensure embedding-function compatibility
    def _needs_reset() -> bool:
        for col in client.list_collections():
            if col.name == "company_docs":
                try:
                    # triggers internal validation
                    _ = col.count()
                    return False
                except Exception:
                    return True
        return False

    if _needs_reset():
        client.delete_collection("company_docs")

    # (2) create or fetch collection with the valid embedding object
    if "company_docs" not in [c.name for c in client.list_collections()]:
        col = client.create_collection(
            "company_docs",
            embedding_function=EMBED,
        )
    else:
        col = client.get_collection("company_docs")  # already OK

    # (3) seed if empty
    if col.count() == 0:
        docs = _seed_docs()
        col.add(
            documents=[d["content"] for d in docs],
            metadatas=[{"ticker": d["ticker"]} for d in docs],
            ids=[d["id"] for d in docs],
        )
    return col
# ───────────────────────────────────────────────────────────────────────────────
