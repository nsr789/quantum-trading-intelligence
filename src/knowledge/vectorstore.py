# ────────────────────────── src/knowledge/vectorstore.py ─────────────────────
"""
Chroma ≥0.5-compatible vector-store helper.

• Supplies a proper SentenceTransformerEmbeddingFunction object
• Deletes legacy collections automatically
• Falls back to in-memory client on read-only file systems (Streamlit Cloud)
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ROOT        = Path(__file__).resolve().parents[2]
STORE_PATH  = ROOT / ".cache" / "chroma"
STORE_PATH.mkdir(parents=True, exist_ok=True)

EMBED = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def _seed() -> List[dict]:
    corpus = [
        ("Apple Inc. designs, manufactures and markets smartphones.", "AAPL"),
        ("Microsoft develops, licenses and supports software.", "MSFT"),
        ("Alphabet (Google) provides internet services and ads.", "GOOGL"),
    ]
    return [{"content": txt, "ticker": tic, "id": f"doc-{i}"} for i, (txt, tic) in enumerate(corpus)]


# -----------------------------------------------------------------------------#
def _client() -> chromadb.api.ClientAPI:
    """Writable persistent client when possible, else in-memory."""
    try:
        return chromadb.PersistentClient(
            path=str(STORE_PATH),
            settings=Settings(allow_reset=True)   # lets us delete incompatible data
        )
    except RuntimeError:                         # read-only FS on Streamlit Cloud
        return chromadb.Client(Settings(allow_reset=True))


def _needs_reset(cli) -> bool:
    """True ↦ legacy collection exists and must be wiped."""
    for col in cli.list_collections():
        if col.name == "company_docs":
            try:             # triggers internal validation
                _ = col.count()
                return False
            except Exception:      # any config mismatch → rebuild
                return True
    return False


def load_vectorstore():
    cli = _client()

    if _needs_reset(cli):
        cli.delete_collection("company_docs")

    # Create or fetch (without passing an incompatible func)
    if "company_docs" not in [c.name for c in cli.list_collections()]:
        col = cli.create_collection("company_docs", embedding_function=EMBED)
    else:
        col = cli.get_collection("company_docs")

    # Seed minimal corpus
    if col.count() == 0:
        docs = _seed()
        col.add(
            documents=[d["content"] for d in docs],
            metadatas=[{"ticker": d["ticker"]} for d in docs],
            ids=[d["id"] for d in docs],
        )
    return col
# ──────────────────────────────────────────────────────────────────────────────
