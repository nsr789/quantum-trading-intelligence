# ────────────────────────── src/knowledge/vectorstore.py ─────────────────────
"""
Chroma helper – now stores a richer fact-sheet per ticker.
"""

from __future__ import annotations
from pathlib import Path

import chromadb
import yfinance as yf
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config.constants import DEFAULT_TICKERS

ROOT        = Path(__file__).resolve().parents[2]
STORE_PATH  = ROOT / ".cache" / "chroma"
STORE_PATH.mkdir(parents=True, exist_ok=True)

EMBED = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


# ───────────────────────── internal helpers ──────────────────────────
def _client() -> chromadb.api.ClientAPI:
    try:                                  # writable on most hosts
        return chromadb.PersistentClient(str(STORE_PATH), Settings(allow_reset=True))
    except RuntimeError:                  # read-only FS – fall back to memory
        return chromadb.Client(Settings(allow_reset=True))


def _fact_sheet(tic: str) -> str | None:
    """Pull multi-line company sheet from yfinance (None on failure)."""
    try:
        info = yf.Ticker(tic).info
    except Exception:
        return None

    if not info:          # network hiccup or invalid ticker
        return None

    summ = info.get("longBusinessSummary") or info.get("shortName") or ""
    sector = info.get("sector", "")
    industry = info.get("industry", "")
    country = info.get("country", "")
    mcap = info.get("marketCap")

    lines = [
        f"{tic} – {summ}",
        f"Sector: {sector}    Industry: {industry}",
        f"Country / HQ: {country}",
        f"Market-cap (USD): {mcap:,}" if mcap else "",
    ]
    return "\n".join(filter(None, lines)).strip() or None


def _seed_many(col: chromadb.Collection, tickers: list[str]) -> None:
    """Add fact-sheets for any *missing* tickers (idempotent)."""
    already = {m["ticker"] for m in col.get(include=["metadatas"])["metadatas"]}
    missing = [t for t in tickers if t not in already]

    docs, metas, ids = [], [], []
    for t in missing:
        sheet = _fact_sheet(t)
        if sheet:
            docs.append(sheet)
            metas.append({"ticker": t})
            ids.append(f"fs-{t}")

    if docs:
        col.add(documents=docs, metadatas=metas, ids=ids)


# ───────────────────────── public loader ─────────────────────────────
def load_vectorstore(ticker: str | None = None) -> chromadb.Collection:
    col = _client().get_or_create_collection("company_docs", embedding_function=EMBED)

    # first-ever run – bootstrap whole DEFAULT_TICKERS set
    if col.count() == 0:
        _seed_many(col, DEFAULT_TICKERS)
    # later runs – ensure *this* ticker exists
    elif ticker:
        _seed_many(col, [ticker.upper()])

    return col
