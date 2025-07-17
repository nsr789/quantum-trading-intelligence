# app/pages/6_🔮_Forecast.py
from __future__ import annotations
import sys, pathlib, datetime as dt

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.express as px
import streamlit as st
from src.config.constants import DEFAULT_TICKERS
from src.ml import PricePredictor

st.markdown("# 🔮 Forecast (LSTM / ONNX)")

ticker = st.selectbox("Ticker", DEFAULT_TICKERS, index=0)
horizon = st.slider("Prediction horizon (days)", 7, 60, 30)
run = st.button("Predict")

if run:
    predictor = PricePredictor()
    forecast = predictor.predict(ticker, horizon=horizon)
    st.metric("Forecasted price", f"${forecast.iloc[-1]:,.2f}")

    hist = forecast.to_frame("forecast")
    hist["date"] = hist.index
    fig = px.line(hist, x="date", y="forecast", title=f"{ticker} price forecast")
    st.plotly_chart(fig, use_container_width=True)
