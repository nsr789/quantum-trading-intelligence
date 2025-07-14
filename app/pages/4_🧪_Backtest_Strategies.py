from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import streamlit as st
import plotly.express as px
import pandas as pd
from src.config.constants import DEFAULT_TICKERS
from src.data.market import get_price_history
from src.strategies import STRATEGY_MAP
from src.strategies.utils import backtest

st.markdown("# 🧪 Strategy Back-tester")

col1, col2 = st.columns(2)
with col1:
    ticker = st.selectbox("Ticker", DEFAULT_TICKERS, 0)
with col2:
    strat_name = st.selectbox("Strategy", list(STRATEGY_MAP))

start = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
run = st.button("Run back-test")

if run:
    price = get_price_history(ticker, start=start)["Close"]
    strategy = STRATEGY_MAP[strat_name]()
    res = backtest(price, strategy)

    st.metric("Cumulative return", f"{res['cum_return']*100:,.1f}%")
    st.metric("Sharpe (daily)", f"{res['sharpe']:.2f}")
    st.metric("Hit-rate", f"{res['hit_rate']*100:.1f}%")

    eq = res["equity"]
    fig = px.line(eq, title="Equity curve")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        pd.DataFrame(
            {"price": price, "position": strategy.generate_signals(price)}
        ).tail(10)
    )
