# ─────────── app/pages/4_🔮_Price_Forecast.py
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
from datetime import datetime, timedelta

from src.data.market import get_price_history
from src.forecast import FORECASTERS
from src.config.constants import DEFAULT_TICKERS

st.markdown("# 🔮 Price Forecast")

with st.sidebar:
    st.markdown("### Parameters")
    ticker = st.selectbox("Ticker", DEFAULT_TICKERS, 0)
    model_name = st.selectbox("Model", list(FORECASTERS))
    horizon = st.slider("Forecast horizon (days)", 1, 30, 5)

    # model-specific knobs (kept simple)
    if model_name == "LSTM":
        lookback = st.slider("Look-back window", 30, 250, 120, 10)
        epochs = st.slider("Epochs", 5, 50, 15, 5)
    elif model_name == "XGBoost":
        lookback = st.slider("Lags", 20, 120, 60, 10)
    run_btn = st.button("Run forecast")

if run_btn:
    # ─────────── fetch history ────────────────────────────────────────────────
    hist = get_price_history(ticker, start=(datetime.now() - timedelta(days=365 * 3)))
    close = hist["Close"].astype("float32")

    # ─────────── forecast ─────────────────────────────────────────────────────
    with st.spinner(f"Running {model_name}…"):
        forecaster = FORECASTERS[model_name]
        if model_name == "LSTM":
            yhat = forecaster(close, horizon=horizon, lookback=lookback, epochs=epochs)
        elif model_name == "XGBoost":
            yhat = forecaster(close, horizon=horizon, n_lags=lookback)
        else:  # linear / ARIMA / SARIMA
            yhat = forecaster(close, horizon=horizon)

    # ─────────── plot ─────────────────────────────────────────────────────────
    fut_idx = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=horizon)
    fig = go.Figure()
    fig.add_scatter(x=close.index, y=close, mode="lines", name="History")
    fig.add_scatter(x=fut_idx, y=yhat, mode="lines+markers", name=f"{model_name} forecast")
    fig.update_layout(height=600, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Last close", f"{close.iloc[-1]:.2f}")
    st.metric("Forecast end", f"{yhat[-1]:.2f}")
else:
    st.info("Set parameters ➜ **Run forecast**.")
