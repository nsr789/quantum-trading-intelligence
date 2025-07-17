from __future__ import annotations
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import streamlit as st
from src.agents.crew_report import generate_report
from src.config.constants import DEFAULT_TICKERS

st.markdown("# 🤖 AI Research Report")
ticker = st.text_input("Ticker", "AAPL")
if st.button("Run AI analysis"):
    with st.spinner("Thinking..."):
        report_md = generate_report(ticker)
    st.markdown(report_md)
