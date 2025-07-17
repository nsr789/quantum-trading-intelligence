# app/main.py
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
    st.markdown("Built by **Satwik Nallamilli** ⚡")

# ───────────────────────────── Main content ──────────────────────────────────
st.title("🧠 Quantum Trading Intelligence")

st.markdown(
    """
Welcome! Choose a tool from the sidebar.  
Here’s what each page does:

| Page | Purpose |
|------|---------|
| **📈 Price Dashboard** | Interactive OHLCV candlestick chart with optional SMA-20 / SMA-50 overlays for *any* US-listed ticker. |
| **📰 News Sentiment** | Fetches the latest headlines, classifies tone *(positive / neutral / negative)*, and shows a sentiment pie. |
| **🤖 AI Research Report** | CrewAI (4 agents) synthesises news, macro, and valuation data into a five-bullet research note — cached to save tokens. |
| **🔮 Forecast** | LSTM model trained on the selected ticker; displays next-day price forecast plus MAE / RMSE metrics. |
| **🧪 Strategy Back-tester** | Runs **rule-based** strategies — *Mean-Reversion* and *Momentum* — then reports return, Sharpe, hit-rate & equity curve. |
| **💬 Company Q&A** | Two-agent RAG chatbot (CrewAI + ChromaDB) that answers arbitrary company questions over an embedded knowledge base. |
"""
)
