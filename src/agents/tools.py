# src/agents/tools.py
"""Utility helpers – headline sentiment now prefers ONNX/XGBoost if present."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Dict, List

import pandas as pd

from src.data.news import fetch_news
from src.ml.xgb_sentiment import HeadlineSentiment  # NEW
from src.utils.logging import get_logger

log = get_logger(module="tools")


# ────────────────────────────────────────────────────────────────────────────
# headline-level sentiment
# ────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_scorer():
    """
    Choose the best available sentiment scorer in this order:

    1. ONNX/XGBoost model trained in `src/ml/xgb_sentiment.py`
    2. 🤗 Transformers small RoBERTa model
    3. TextBlob polarity
    """
    # 1️⃣ ONNX scorer (always available because a tiny fallback model is exported)
    try:
        return ("onnx", HeadlineSentiment())
    except Exception as exc:  # pragma: no cover
        log.warning("ONNX headline sentiment unavailable → %s", exc)

    # 2️⃣ 🤗
    try:
        transformers = importlib.import_module("transformers")  # type: ignore
        pipe = transformers.pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )
        return ("hf", pipe)
    except Exception as exc:  # pragma: no cover
        log.warning("HF pipeline unavailable → %s", exc)

    # 3️⃣ TextBlob
    try:
        from textblob import TextBlob  # type: ignore

        def _tb(text: str) -> float:
            return TextBlob(text).sentiment.polarity

        return ("tb", _tb)
    except ModuleNotFoundError:  # pragma: no cover
        return ("none", lambda _: 0.0)


def _score_titles(titles: List[str]) -> List[float]:
    mode, scorer = _get_scorer()

    if mode == "onnx":
        return [(p - 0.5) * 2 for p in scorer.score(titles)]  # map [0,1] → [-1,1]
    if mode == "hf":
        mapping = {"negative": -1, "neutral": 0, "positive": 1}
        return [mapping[r["label"].lower()] * r["score"] for r in scorer(titles)]
    if mode == "tb":
        return [scorer(t) for t in titles]
    return [0.0 for _ in titles]


def company_news_sentiment(ticker: str, limit: int = 15) -> Dict[str, float]:
    df: pd.DataFrame = fetch_news(ticker, limit=limit)
    if df.empty:
        log.warning("No headlines for %s – returning neutral", ticker)
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    scores = _score_titles(df["title"].astype(str).tolist())
    pos = sum(s > 0.1 for s in scores) / len(scores)
    neg = sum(s < -0.1 for s in scores) / len(scores)
    neu = 1 - pos - neg
    return {"positive": round(pos, 2), "negative": round(neg, 2), "neutral": round(neu, 2)}
