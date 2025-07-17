from __future__ import annotations  # ← must be FIRST line in file

# --- project bootstrap (makes `src` importable when Streamlit's CWD = /app) --
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------------------------------------------

import re
from datetime import datetime, timezone
from typing import Final

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta

from src.config.constants import DEFAULT_TICKERS
from src.data.market import get_price_history

# ────────────────────────── Helper: period string → Timestamp ──────────────
_PERIOD_RE: Final = re.compile(r"(?P<num>\d+)(?P<unit>d|w|mo|y)$")


def start_from_period(period: str) -> pd.Timestamp:
    """Convert '3mo', '2y', '10d', '4w' → Timestamp.now() minus that span."""
    m = _PERIOD_RE.fullmatch(period)
    if not m:
        raise ValueError(f"Invalid period string: {period}")

    num = int(m["num"])
    unit = m["unit"]
    now = pd.Timestamp.now(tz=timezone.utc)

    if unit == "d":
        return now - pd.Timedelta(days=num)
    if unit == "w":
        return now - pd.Timedelta(weeks=num)
    if unit == "mo":
        return now - relativedelta(months=num)
    if unit == "y":
        return now - relativedelta(years=num)
    raise AssertionError("unreachable")


# ─────────────────────────────── Streamlit UI ───────────────────────────────
st.markdown("# 📈 Price Dashboard")

with st.sidebar:
    st.markdown("### Parameters")
    ticker = st.selectbox("Ticker", DEFAULT_TICKERS, index=0)
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    show_sma20 = st.checkbox("Show SMA-20", value=True)
    show_sma50 = st.checkbox("Show SMA-50", value=True)
    run_btn = st.button("Update chart")

if run_btn:
    start_date = start_from_period(period)

    df = get_price_history(ticker, start=start_date, end=datetime.now(timezone.utc))[
        ["Open", "High", "Low", "Close", "Volume"]
    ].copy()

    if show_sma20:
        df["SMA20"] = df["Close"].rolling(20).mean()
    if show_sma50:
        df["SMA50"] = df["Close"].rolling(50).mean()

    # ── Plotly chart ───────────────────────────────────────────────────────
    fig = go.Figure()

    # down-sample to ≤300 points for speed
    sample = max(1, len(df) // 300)
    cs = df.iloc[::sample]

    fig.add_trace(
        go.Candlestick(
            x=cs.index,
            open=cs["Open"],
            high=cs["High"],
            low=cs["Low"],
            close=cs["Close"],
            name="OHLC",
            showlegend=False,
        )
    )

    if show_sma20:
        fig.add_scatter(x=df.index, y=df["SMA20"], mode="lines", name="SMA 20")
    if show_sma50:
        fig.add_scatter(x=df.index, y=df["SMA50"], mode="lines", name="SMA 50")

    fig.update_layout(height=600, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # table
    st.dataframe(df.tail(10))
else:
    st.info("Choose parameters in the sidebar and press **Update chart**.")
