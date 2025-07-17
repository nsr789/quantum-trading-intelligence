# ─────────────────────────────────────────────────────────────── app/pages/6_🔮_Forecast.py
from __future__ import annotations

# --- project bootstrap (makes `src` importable when Streamlit’s CWD = /app) ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta

from src.data.market import get_price_history
from src.forecast.lstm_forecaster import lstm_forecast
from src.config.constants import DEFAULT_TICKERS

st.markdown("# 🔮 Price Forecast")

with st.sidebar:
    st.markdown("### Parameters")
    ticker = st.selectbox("Ticker", DEFAULT_TICKERS, index=0)
    horizon = st.slider("Forecast horizon (trading days)", 1, 30, 5)
    lookback = st.slider("Look-back window", 30, 250, 120, step=10)
    epochs = st.slider("Epochs", 5, 50, 15, step=5)
    run_btn = st.button("Run forecast")

if run_btn:
    # ── fetch history ─────────────────────────────────────────────────────────
    hist = get_price_history(ticker, start=(datetime.now() - timedelta(days=lookback*3)))
    close = hist["Close"].astype("float32")

    # ── forecast ──────────────────────────────────────────────────────────────
    with st.spinner("Training LSTM..."):
        yhat = lstm_forecast(close, horizon=horizon, lookback=lookback, epochs=epochs)

    # ── plot ──────────────────────────────────────────────────────────────────
    fut_idx = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=horizon)
    fig = go.Figure()
    fig.add_scatter(x=close.index, y=close, mode="lines", name="History")
    fig.add_scatter(x=fut_idx, y=yhat, mode="lines+markers", name="Forecast")
    fig.update_layout(height=600, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Last close", f"{close.iloc[-1]:.2f}")
    st.metric("Forecast end", f"{yhat[-1]:.2f}")
else:
    st.info("Configure parameters in the sidebar and press **Run forecast**.")
