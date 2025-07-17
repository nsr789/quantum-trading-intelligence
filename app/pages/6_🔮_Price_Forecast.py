# app/pages/6_🔮_Price_Forecast.py
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import streamlit as st
import plotly.express as px
import pandas as pd

from src.ml.predictor import PricePredictor
from src.config.constants import DEFAULT_TICKERS

st.markdown("# 🔮 Price Forecast")

col1, col2, col3 = st.columns(3)
with col1:
    tic = st.selectbox("Ticker", DEFAULT_TICKERS, 0)
with col2:
    win = st.slider("Look-back window", 10, 60, 30, 5)
with col3:
    eng = st.selectbox("Model", ["linear", "xgb", "lstm"], index=0)

if st.button("Predict"):
    pr = PricePredictor(tic, window=win, engine=eng)
    nxt = pr.predict_next_close()
    ser = pr.predict_series()
    st.metric("Next close", f"${nxt:,.2f}")
    fig = px.line(ser, title="5-day forecast")
    st.plotly_chart(fig, use_container_width=True)
