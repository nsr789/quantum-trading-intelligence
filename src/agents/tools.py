# src/agents/tools.py
"""Utility helpers used by multi-agent pipelines (sentiment, embeddings, etc.)."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Dict, List

import pandas as pd

from src.data.news import fetch_news
from src.utils.logging import get_logger

log = get_logger(module="tools")

# --------------------------------------------------------------------------- #
# 📰 News headline sentiment
# --------------------------------------------------------------------------- #
_HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # tiny + robust


@lru_cache(maxsize=1)
def _load_sentiment_model():
    """
    Lazy-load a sentiment-analysis pipeline.

    1. Try 🤗 Transformers (preferred - higher quality).
    2. Fallback → TextBlob (always available via std-install).
    """
    try:
        transformers = importlib.import_module("transformers")  # type: ignore
        pipeline = transformers.pipeline("sentiment-analysis", model=_HF_MODEL)
        log.info("Loaded Hugging Face sentiment pipeline (%s)", _HF_MODEL)
        return ("hf", pipeline)
    except Exception as exc:  # pragma: no cover
        log.warning("HF pipeline unavailable – falling back to TextBlob (%s)", exc)

    # ----- fallback -----
    try:
        from textblob import TextBlob  # type: ignore

        def _tb_sent(text: str) -> float:
            return TextBlob(text).sentiment.polarity

        log.info("Loaded TextBlob fallback sentiment analyzer.")
        return ("tb", _tb_sent)
    except ModuleNotFoundError as exc:  # pragma: no cover
        log.error("TextBlob not installed (%s) – sentiment disabled.", exc)
        return ("none", lambda _: 0.0)  # neutral everywhere


def _score_headlines(headlines: List[str]) -> List[float]:
    """Return polarity scores in range [-1, 1] for a list of headlines."""
    mode, model = _load_sentiment_model()

    if mode == "hf":
        # Hugging Face returns dicts like {"label": "positive", "score": 0.97}
        mapping = {"negative": -1, "neutral": 0, "positive": 1}

        results = model(headlines, truncation=True)
        return [mapping[r["label"].lower()] * r["score"] for r in results]

    # TextBlob or neutral-only fallback
    return [model(h) for h in headlines]


def company_news_sentiment(ticker: str, limit: int = 15) -> Dict[str, float]:
    """
    Compute sentiment share ({positive, negative, neutral}) for latest headlines.

    The function is *deterministic* for a given set of titles so it remains
    cache-friendly via `src.data.cache.cached`.
    """
    df: pd.DataFrame = fetch_news(ticker, limit=limit)
    if df.empty or "title" not in df:
        log.warning("No headlines found for %s – returning neutral distribution", ticker)
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    scores = _score_headlines(df["title"].astype(str).tolist())

    pos = sum(1 for s in scores if s > 0.1) / len(scores)
    neg = sum(1 for s in scores if s < -0.1) / len(scores)
    neu = 1.0 - pos - neg

    return {
        "positive": round(pos, 2),
        "negative": round(neg, 2),
        "neutral": round(neu, 2),
    }
