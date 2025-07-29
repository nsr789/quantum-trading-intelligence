# app/main.py

import os
os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "poll"

# --- project bootstrap (makes `src` importable) ------------------------------
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -----------------------------------------------------------------------------


import streamlit as st
from src.config.settings import settings

# ────────────────────────────── Page config ──────────────────────────────────
st.set_page_config(
    page_title="Quantum Trading Intelligence",
    page_icon="🧠",
    layout="wide",
)

# ───────────────────────────── Sidebar (settings) ────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️  Runtime keys")
    st.markdown(
        f"""
**OpenAI key set:** {bool(settings.OPENAI_API_KEY)}  
**Groq key set:** {bool(settings.GROQ_API_KEY)}  
**NewsAPI key set:** {bool(settings.NEWS_API_KEY)}
"""
    )
    st.write("---")
    st.markdown(
        """
**Developed by Satwik Nallamilli** ⚡  
[LinkedIn](https://www.linkedin.com/in/satwik-nallamilli-358397218/)  
[GitHub](https://github.com/nsr789)  
📧 ssn9@illinois.edu
"""
    )

# ───────────────────────────── Main content ──────────────────────────────────
st.title("🧠 Quantum Trading Intelligence")

st.markdown(
    """
Welcome! Choose a tool from the sidebar.  
Here’s what each page does:

| Page | Purpose |
|------|---------|
| **📈 Price Dashboard** | Interactive OHLC-V candlestick chart with optional SMA-20 / SMA-50 overlays. |
| **📰 News Sentiment** | Scrapes the most-recent headlines **and full-text** articles, runs them through FinBERT (transformer) to classify tone *(positive / neutral / negative)*, then shows a sentiment pie. |
| **🤖 AI Research Report** | A 4-agent CrewAI pipeline blends FinBERT sentiment, macro data, and valuation metrics into a concise 5-bullet research note — cached to save tokens. |
| **🔮 Price Forecast** | Pick a model (LSTM · Linear Reg · ARIMA / SARIMA · XGBoost) and generate a forward price path plotted alongside recent history. |
| **🧪 Strategy Back-tester** | Evaluate **rule-based** trading strategies — *Mean-Reversion* and *SMA-Momentum* — and view Sharpe, hit-rate, and equity curve. |
| **💬 Company Q&A** | Two-agent RAG chatbot (CrewAI + ChromaDB) that answers detailed company questions using an embedded knowledge base. |
"""
)
