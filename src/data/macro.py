"""Macro-economic data via FRED (fallback to static)."""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

try:
    from fredapi import Fred  # type: ignore
except ImportError:  # pragma: no cover
    Fred = None  # pyright: ignore

from src.config.settings import settings
from src.data.cache import cached
from src.utils.logging import get_logger

log = get_logger(module="macro")


@cached
def get_fred_series(series_id: str, start: str | date = "2015-01-01") -> pd.Series:
    """Return a pandas Series of a FRED time-series."""
    if not (settings.FRED_API_KEY and Fred):
        raise RuntimeError("FRED API key missing or fredapi not installed.")
    fred = Fred(api_key=settings.FRED_API_KEY)
    data = fred.get_series(series_id, observation_start=start)
    if data.empty:
        raise ValueError(f"No data for FRED series {series_id}")
    data.name = series_id
    return data


def get_unemployment_rate() -> Optional[float]:
    """Return latest US unemployment rate or None if API unavailable."""
    try:
        return float(get_fred_series("UNRATE").dropna().iloc[-1])
    except Exception as exc:  # noqa: BLE001
        log.warning(f"macro UNRATE unavailable: {exc}")
        return None
