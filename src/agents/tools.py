# ── src/agents/tools.py ───────────────────────────────────────────────────────
"""
Utilities that agents can call (news-sentiment, etc.).

Changes v2
* Gracefully handle environments where TextBlob data corpora
  haven't been downloaded yet (Streamlit Cloud, CI, etc.).
* Keeps public function signatures unchanged – no downstream changes.
"""
from __future__ import annotations
from collections import Counter
from typing import Dict, List

# ── sentiment helper ----------------------------------------------------------
try:
    from textblob import TextBlob
except ModuleNotFoundError as _err:  # pragma: no cover – tests install it
    raise ImportError(
        "The 'textblob' package is required. "
        "Add `textblob` to requirements.txt and run `pip install -r requirements.txt`."
    ) from _err


def _blob_polarity(text: str) -> float:
    """
    Return polarity in [-1, 1] but fail *silently* if corpora missing
    (TextBlob uses pattern library corpora – absent on many CI boxes).
    """
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:  # noqa: BLE001
        # Fallback: neutral
        return 0.0


# ── public API ----------------------------------------------------------------
def company_news_sentiment(ticker: str, limit: int = 20) -> Dict[str, float]:
    """
    Very small demo helper:
    * fetch_news is defined elsewhere and returns List[str] titles
    * classifies into positive / negative / neutral using TextBlob polarity
    """
    from src.data.news import fetch_news  # local import to avoid heavy deps at import-time

    headlines: List[str] = fetch_news(ticker, limit).get("title", [])  # type: ignore[arg-type]

    buckets: Counter[str] = Counter()
    for h in headlines:
        p = _blob_polarity(h)
        if p > 0.05:
            buckets["positive"] += 1
        elif p < -0.05:
            buckets["negative"] += 1
        else:
            buckets["neutral"] += 1

    total = sum(buckets.values()) or 1
    return {k: v / total for k, v in buckets.items()}
