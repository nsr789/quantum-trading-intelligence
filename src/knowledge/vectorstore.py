# ────────────────────────── src/knowledge/vectorstore.py ─────────────────────
"""
Chroma ≥ 0.5 helper – bullet-proof against legacy collections on Streamlit Cloud
and always provides a valid SentenceTransformerEmbeddingFunction.
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
        ("Microsoft develops, licenses and supports software products.", "MSFT"),
        ("Alphabet (Google) provides internet services and ads.", "GOOGL"),
    ]
    return [{"content": txt, "ticker": tic, "id": f"doc-{i}"} for i, (txt, tic) in enumerate(corpus)]


def _client() -> chromadb.api.ClientAPI:
    """Writable persistent client when possible, else in-memory (read-only FS)."""
    try:
        return chromadb.PersistentClient(
            path=str(STORE_PATH),
            settings=Settings(allow_reset=True),
        )
    except RuntimeError:                      # e.g. Streamlit Community Cloud
        return chromadb.Client(Settings(allow_reset=True))


def load_vectorstore():
    cli = _client()

    # ----- robust creation ----------------------------------------------------
    try:
        col = cli.get_or_create_collection("company_docs", embedding_function=EMBED)
    except AttributeError:
        # A collection created with an old plain-function embedding exists → nuke
        cli.delete_collection("company_docs")
        col = cli.create_collection("company_docs", embedding_function=EMBED)
    # -------------------------------------------------------------------------

    if col.count() == 0:              # initial seed (only 3 mini docs)
        docs = _seed()
        col.add(
            documents=[d["content"] for d in docs],
            metadatas=[{"ticker": d["ticker"]} for d in docs],
            ids=[d["id"] for d in docs],
        )
    return col
# ──────────────────────────────────────────────────────────────────────────────
