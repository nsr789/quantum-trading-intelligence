"""Tiny cached scraper for article body text."""
from __future__ import annotations
from functools import lru_cache

import newspaper  # pip install newspaper3k

@lru_cache(maxsize=512)
def extract_text(url: str) -> str:
    """Return plain-text article body or empty string on failure."""
    if not url:
        return ""
    try:
        art = newspaper.Article(url, language="en")
        art.download(); art.parse()
        return art.text or ""
    except Exception:   # pragma: no cover
        return ""
