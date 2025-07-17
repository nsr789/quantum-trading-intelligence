"""Static constants used across the project."""

from __future__ import annotations

from datetime import timedelta

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA",
    "NVDA", "META", "AVGO", "NFLX", "COST",
    "PLTR", "ASML", "CSCO", "TMUS", "AMD",
    "LIN", "AZN", "INTU", "TXN", "BKNG"
]
SUPPORTED_INTERVALS = ["1d", "1wk", "1mo"]
MAX_CANDLE_POINTS = 5_000  # plotly down-sample guard
SENTIMENT_ROLLING_WINDOW = 7  # days
BACKTEST_START_DELTA = timedelta(days=730)  # 2 years default look-back
FORECAST_LOOKBACK = 120   # days
FORECAST_HORIZON  = 5     # trading days
