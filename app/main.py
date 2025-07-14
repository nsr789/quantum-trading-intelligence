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
    st.markdown("## ⚙️ Settings")
    st.markdown(
        f"""
*OpenAI key set:* **{bool(settings.OPENAI_API_KEY)}**  
*Groq key set:* **{bool(settings.GROQ_API_KEY)}**  
*NewsAPI key set:* **{bool(settings.NEWS_API_KEY)}**
"""
    )
    st.write("---")
    st.markdown("Developed by **Satwik Nallamilli** ⚡")

# ───────────────────────────── Main content ──────────────────────────────────
st.title("🧠 Quantum Trading Intelligence")

st.markdown(
    """
Welcome! Pick a tool from the left-hand sidebar.  
Below is a quick overview of each page/folder:

| Page | What it does |
|------|--------------|
| **📈 Price Dashboard** | Interactive OHLC candlestick chart with optional SMA-20 / SMA-50 overlays for *any* US-listed ticker. |
| **📰 News Sentiment** | Pulls the latest headlines, classifies tone (positive / negative / neutral) and shows a sentiment pie. |
| **🤖 AI Research Report** | Multi-agent CrewAI pipeline that combines news, macro, and valuation data into a five-bullet research note — cached to save tokens. |
| **🧪 Strategy Back-tester** | Runs vectorised mean-reversion or SMA-momentum strategies on your chosen ticker, spitting out returns, Sharpe, hit-rate & equity curve. |
| **💬 Company Q&A** | Two-agent RAG chatbot (CrewAI + ChromaDB) that answers arbitrary questions using an embedded knowledge base. |

"""
)
