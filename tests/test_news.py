import os, pandas as pd, pytest
from src.data.news import fetch_news

def test_fetch_news():
    df = fetch_news("Apple", limit=5)
    if df.empty:
        pytest.skip("No NewsAPI key and yfinance fallback returned no news")
    assert {"title", "source"}.issubset(df.columns)
