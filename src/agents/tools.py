"""Utility agents / lightweight analytics."""
from __future__ import annotations

import pandas as pd
from textblob import TextBlob

from src.data.news     import fetch_news
from src.data.scrape   import extract_text
from src.agents.sentiment import transformer_sentiment


def company_news_sentiment(
    ticker: str,
    limit: int = 15,
    full_article: bool = False,
) -> dict[str, float]:
    """
    Return sentiment share dict **{positive, negative, neutral}**.

    Parameters
    ----------
    ticker        : company ticker
    limit         : number of headlines
    full_article  : if True, scrape each url and run FinBERT; otherwise quick
                    headline polarity via TextBlob (fast, zero-cost).
    """
    df = fetch_news(ticker, limit=limit)

    if full_article:
        # -------- transformer path -----------------------------------------
        bodies = [
            extract_text(u) or t      # fall back to title if scrape fails
            for t, u in zip(df["title"], df["url"])
        ]
        labels = pd.Series(transformer_sentiment(bodies))
        share  = labels.value_counts(normalize=True)
        return {k: round(share.get(k, 0.0), 2) for k in ["positive", "negative", "neutral"]}

    # -------- headline fallback --------------------------------------------
    polar = df["title"].map(lambda t: TextBlob(str(t)).sentiment.polarity)
    pos = (polar >  0.10).mean()
    neg = (polar < -0.10).mean()
    neu = 1 - pos - neg
    return {"positive": round(pos, 2), "negative": round(neg, 2), "neutral": round(neu, 2)}
