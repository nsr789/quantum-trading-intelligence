# ── src/data/news.py ───────────────────────────────────────────────────────────
"""News & Reddit sentiment helpers – always return a valid DataFrame."""

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


# ──────────────────────────────  Headlines  ───────────────────────────────────
@cached
def fetch_news(query: str, limit: int = 20) -> pd.DataFrame:
    """Return DataFrame[date,title,source] (never empty)."""
    # 1️⃣  NewsAPI (if key configured) -----------------------------------------
    if settings.NEWS_API_KEY:
        params = dict(
            q=query,
            apiKey=settings.NEWS_API_KEY,
            pageSize=limit,
            sortBy="publishedAt",
            language="en",
        )
        try:
            r = requests.get(NEWS_ENDPOINT, params=params, timeout=15)
            r.raise_for_status()
            arts = r.json().get("articles", [])
            if arts:
                return pd.DataFrame(
                    {
                        "date": [a["publishedAt"] for a in arts],
                        "title": [a["title"] for a in arts],
                        "source": [a["source"]["name"] for a in arts],
                    }
                )
        except Exception as exc:  # pragma: no cover
            log.warning("NewsAPI fetch failed: {}", exc)

    # 2️⃣  Yahoo Finance fallback ---------------------------------------------
    rows: list[dict] = []
    for n in (yf.Ticker(query).news or [])[:limit]:
        ts = (
            n.get("providerPublishTime")
            or n.get("provider_publish_time")
            or n.get("pubDate")
        )
        try:
            dt = _dt.datetime.fromtimestamp(ts) if ts else _dt.datetime.utcnow()
        except Exception:  # badly-formed timestamp
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

    # 3️⃣  Guaranteed synthetic row – keeps tests deterministic ----------------
    log.warning("No live news found – returning synthetic placeholder row.")
    return pd.DataFrame(
        {
            "date": [_dt.datetime.utcnow()],
            "title": [f"No recent news for {query}"],
            "source": ["synthetic"],
        }
    )


# ──────────────────────────────  Reddit  ──────────────────────────────────────
@cached
def fetch_reddit(subreddit: str = "wallstreetbets", limit: int = 50) -> List[str]:
    """Return list of post titles (requires Reddit creds)."""
    if not (settings.REDDIT_CLIENT_ID and praw):
        log.warning("Reddit creds missing – subreddit fetch skipped.")
        return []

    reddit = praw.Reddit(
        client_id=settings.REDDIT_CLIENT_ID,
        client_secret=settings.REDDIT_CLIENT_SECRET,
        user_agent=settings.REDDIT_USER_AGENT,
    )
    return [p.title for p in reddit.subreddit(subreddit).hot(limit=limit)]
