"""Market-data helpers.

* 100 % local dependencies → yfinance (always works, no key required)
* Optionally enrich with AlphaVantage / Finnhub if their keys are set.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Dict, List

import pandas as pd
import yfinance as yf


try:
    from alpha_vantage.fundamentaldata import FundamentalData  # type: ignore
except ImportError:  # pragma: no cover
    FundamentalData = None  # pyright: ignore

try:
    import finnhub  # type: ignore
except ImportError:  # pragma: no cover
    finnhub = None  # pyright: ignore

from src.config.settings import settings
from src.data.cache import cached
from src.utils.logging import get_logger

log = get_logger(module="market")
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# ── price history ─────────────────────────────────────────────────────────────
@cached
def get_price_history(
    ticker: str,
    start: str | date = "2020-01-01",
    end: str | date | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Return OHLCV dataframe (uses yfinance)."""
    end = end or date.today()
    df = yf.download(
        ticker, start=start, end=end, interval=interval, group_by="column", auto_adjust=False
    )
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, level=1, axis=1)  # keep single-level columns
    if df.empty:
        raise ValueError(f"Empty price history for {ticker}")
    df.index.name = "date"
    return df


# ── fundamentals (multi-provider) ─────────────────────────────────────────────
@cached
def get_fundamentals(ticker: str) -> Dict[str, float | str]:
    """Return lightweight fundamentals dict.

    Order of preference:
    1. AlphaVantage (`overview` endpoint)    – rich & fast
    2. Finnhub (`company_basic_financials`) – also good
    3. yfinance `.info`                     – public fallback
    """
    if settings.ALPHAVANTAGE_API_KEY and FundamentalData:
        fd = FundamentalData(settings.ALPHAVANTAGE_API_KEY)
        data, _ = fd.get_company_overview(ticker)
        if data:
            return {
                "MarketCap": float(data["MarketCapitalization"]),
                "PERatio": float(data["PERatio"]),
                "EPS": float(data["EPS"]),
                "DividendYield": float(data["DividendYield"] or 0),
                "ProfitMargin": float(data["ProfitMargin"] or 0),
            }

    if settings.FINNHUB_API_KEY and finnhub:
        client = finnhub.Client(api_key=settings.FINNHUB_API_KEY)
        basic = client.company_basic_financials(ticker, "all").get("metric", {})
        if basic:
            return {
                "MarketCap": basic.get("marketCapitalization", 0.0),
                "PERatio": basic.get("peBasicExclExtraTTM", 0.0),
                "EPS": basic.get("epsInclExtraItemsTTM", 0.0),
                "DividendYield": basic.get("dividendYieldIndicatedAnnual", 0.0),
                "ProfitMargin": basic.get("netProfitMarginTTM", 0.0),
            }

    # fallback
    info = yf.Ticker(ticker).info
    return {
        "MarketCap": info.get("marketCap"),
        "PERatio": info.get("trailingPE"),
        "EPS": info.get("trailingEps"),
        "DividendYield": info.get("dividendYield"),
        "ProfitMargin": info.get("profitMargins"),
    }


# ── async batch helper ────────────────────────────────────────────────────────
async def _fetch_one(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, get_price_history, ticker, start, end, interval
    )


async def get_price_batch(
    tickers: List[str],
    start: str | date = "2023-01-01",
    end: str | date | None = None,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Fetch multiple tickers in parallel (thread-pool)."""
    coros = [_fetch_one(t, start, end, interval) for t in tickers]
    data = await asyncio.gather(*coros, return_exceptions=True)
    return {t: d for t, d in zip(tickers, data) if isinstance(d, pd.DataFrame)}
