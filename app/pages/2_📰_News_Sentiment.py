from __future__ import annotations
# --- bootstrap ---------------------------------------------------------------
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
# ----------------------------------------------------------------------------- 

import streamlit as st
import plotly.express as px
from src.data.news import fetch_news
from src.agents.tools import company_news_sentiment
from src.config.constants import DEFAULT_TICKERS

st.markdown("# 📰 News Sentiment")

ticker = st.selectbox("Ticker", DEFAULT_TICKERS, index=0)
limit = st.slider("Headlines", 10, 50, 20)
if st.button("Analyse"):
    df = fetch_news(ticker, limit)
    if df.empty:
        st.warning("No headlines found.")
    else:
        st.write("Latest headlines", df[["date", "title"]])
        sent = company_news_sentiment(ticker, limit)
        st.json(sent)
        fig = px.pie(values=list(sent.values()), names=list(sent.keys()), title="Sentiment Share")
        st.plotly_chart(fig, use_container_width=True)
