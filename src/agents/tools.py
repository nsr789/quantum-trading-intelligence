from __future__ import annotations
from textblob import TextBlob
from src.data.news import fetch_news

def company_news_sentiment(ticker: str, limit: int = 15) -> dict[str, float]:
    """Return {pos, neg, neu} sentiment proportions from latest headlines."""
    df = fetch_news(ticker, limit=limit)
    polarities = df["title"].map(lambda t: TextBlob(str(t)).sentiment.polarity)
    pos = (polarities > 0.1).mean()
    neg = (polarities < -0.1).mean()
    neu = 1 - pos - neg
    return {"positive": round(pos, 2), "negative": round(neg, 2), "neutral": round(neu, 2)}
