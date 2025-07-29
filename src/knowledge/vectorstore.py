"""
Vector-store helper (Chroma ≥ 0.5)
• one collection:  company_docs
• embedding:       MiniLM-L6-v2  (∼384-d)
• auto-bootstrap:  yfinance description if ticker unseen
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb
import yfinance as yf
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ROOT = Path(__file__).resolve().parents[2]
STORE_PATH = ROOT / ".cache" / "chroma"
STORE_PATH.mkdir(parents=True, exist_ok=True)

EMBED = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def _client() -> chromadb.api.ClientAPI:
    """Writable persistent client when possible, else in-memory fallback."""
    try:
        return chromadb.PersistentClient(path=str(STORE_PATH), settings=Settings(allow_reset=True))
    except RuntimeError:  # Read-only FS (e.g. Streamlit Cloud)
        return chromadb.Client(Settings(allow_reset=True))


def _bootstrap(col: chromadb.Collection, ticker: str) -> None:
    """If docs for *ticker* missing → pull short description from yfinance."""
    if col.count(where={"ticker": ticker}) > 0:
        return

    try:
        info = yf.Ticker(ticker).info
        desc = info.get("longBusinessSummary") or info.get("shortName")
        if not desc:
            return
    except Exception:  # network hiccup → silent fail
        return

    col.add(
        documents=[desc],
        metadatas=[{"ticker": ticker}],
        ids=[f"seed-{ticker}"],
    )


def load_vectorstore(ticker: str | None = None) -> chromadb.Collection:
    cli = _client()

    # robust create / migrate
    try:
        col = cli.get_or_create_collection("company_docs", embedding_function=EMBED)
    except AttributeError:  # legacy collection with plain-fn embeddings
        cli.delete_collection("company_docs")
        col = cli.create_collection("company_docs", embedding_function=EMBED)

    # initial seed (tiny demo corpus)
    if col.count() == 0:
        col.add(
            documents=[
                "Apple Inc. designs, manufactures and markets smartphones.",
                "Microsoft develops, licenses and supports software products.",
                "Alphabet (Google) provides internet services and ads.",
            ],
            metadatas=[{"ticker": t} for t in ("AAPL", "MSFT", "GOOGL")],
            ids=[f"doc-{i}" for i in range(3)],
        )

    # ensure docs for the requested ticker exist
    if ticker:
        _bootstrap(col, ticker.upper())

    return col
