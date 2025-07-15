"""News & Reddit sentiment sources."""

from __future__ import annotations

import datetime as _dt
from typing import List

import pandas as pd
import requests
import yfinance as yf

try:
    import praw  # type: ignore
except ImportError:  # pragma: no cover
    praw = None  # pyright: ignore

from src.config.settings import settings
from src.data.cache import cached
from src.utils.logging import get_logger

log = get_logger(module="news")

NEWS_ENDPOINT = "https://newsapi.org/v2/everything"


# ──────────────────────────────────────────────────────────────────────────────
@cached
def fetch_news(query: str, limit: int = 20) -> pd.DataFrame:
    """Return **always-valid** DataFrame[date, title, source] for recent news.

    Fallback chain:
    1. NewsAPI  (if key provided)
    2. yfinance headlines (robust to missing fields & API drift)
    3. Synthetic placeholder row (keeps downstream tests deterministic)
    """
    # 1️⃣  NewsAPI ----------------------------------------------------------------
    if settings.NEWS_API_KEY:
        params = {
            "q": query,
            "apiKey": settings.NEWS_API_KEY,
            "pageSize": limit,
            "sortBy": "publishedAt",
            "language": "en",
        }
        resp = requests.get(NEWS_ENDPOINT, params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("articles", [])
        if items:
            return pd.DataFrame(
                {
                    "date": [i["publishedAt"] for i in items],
                    "title": [i["title"] for i in items],
                    "source": [i["source"]["name"] for i in items],
                }
            )

    # 2️⃣  yfinance fallback ------------------------------------------------------
    # Yahoo occasionally drops/renames keys → use .get() & defaults.
    news = (yf.Ticker(query).news or [])[:limit]
    if news:
        rows: list[dict] = []
        for n in news:
            ts = (
                n.get("providerPublishTime")            # old key
                or n.get("provider_publish_time")       # newer snake-case key
                or n.get("pubDate")                     # very old RSS key
            )
            try:
                dt = _dt.datetime.fromtimestamp(ts) if ts else _dt.datetime.utcnow()
            except Exception:
                dt = _dt.datetime.utcnow()

            rows.append(
                {
                    "date": dt,
                    "title": n.get("title", ""),
                    "source": n.get("publisher", "") or n.get("source", ""),
                }
            )
        if rows:
            return pd.DataFrame(rows)

    # 3️⃣  Synthetic placeholder --------------------------------------------------
    log.warning("No live news found – returning synthetic placeholder row.")
    return pd.DataFrame(
        {
            "date": [_dt.datetime.utcnow()],
            "title": [f"No recent news for {query}"],
            "source": ["synthetic"],
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
@cached
def fetch_reddit(subreddit: str = "wallstreetbets", limit: int = 50) -> List[str]:
    """Return list of post titles (requires Reddit creds)."""
    if not (settings.REDDIT_CLIENT_ID and praw):
        log.warning("Reddit creds missing – skipping subreddit fetch.")
        return []

    reddit = praw.Reddit(
        client_id=settings.REDDIT_CLIENT_ID,
        client_secret=settings.REDDIT_CLIENT_SECRET,
        user_agent=settings.REDDIT_USER_AGENT,
    )
    posts = reddit.subreddit(subreddit).hot(limit=limit)
    return [p.title for p in posts]
