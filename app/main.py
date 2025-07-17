# app/main.py
# ── project bootstrap (makes `src` importable when Streamlit sets CWD=/app) ──
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------------------------------------------

import streamlit as st
from src.config.settings import settings

# ────────────────────────────── Page config ────────────────────────────────
st.set_page_config(
    page_title="Quantum Trading Intelligence",
    page_icon="🧠",
    layout="wide",
)

# ───────────────────────────── Sidebar (settings) ──────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Runtime Settings")
    st.markdown(
        f"""
*OpenAI key set:* **{bool(settings.OPENAI_API_KEY)}**  
*Groq key set:* **{bool(settings.GROQ_API_KEY)}**  
*NewsAPI key set:* **{bool(settings.NEWS_API_KEY)}**
"""
    )
    st.write("---")
    st.markdown(
        "Built by **Satwik Nallamilli** · "
        "[GitHub](https://github.com/nsr789) · "
        "[LinkedIn](https://www.linkedin.com/in/satwik-nallamilli-358397218/)"
    )

# ───────────────────────────── Main content ────────────────────────────────
st.title("🧠 Quantum Trading Intelligence")

st.markdown(
    """
A lightweight research cockpit that now **leverages a transformer-based sentiment
model (RoBERTa via 🤗 Transformers)** and is wired for upcoming LSTM / XGBoost
price-forecast modules exported to **ONNX** for edge deployment.

| Page | Purpose |
|------|---------|
| **📈 Price Dashboard** | Interactive OHLC candlestick chart with optional SMA-20 / SMA-50 overlays for **any** US-listed ticker. |
| **📰 News Sentiment** | Pulls the latest headlines and classifies tone using a **cardiffnlp/twitter-roberta-base-sentiment-latest** model (falls back to TextBlob if offline). |
| **🤖 AI Research Report** | 4-agent CrewAI pipeline that fuses news tone, macro data and valuation metrics into a 5-bullet research note — cached to spare tokens. |
| **🧪 Strategy Back-tester** | Vectorised mean-reversion & SMA momentum strategies with equity curve, returns, Sharpe & hit-rate. |
| **💬 Company Q&A** | Two-agent RAG chatbot (CrewAI + ChromaDB) answering free-form company questions from an embedded knowledge base. |

---

**What’s next?**  
The codebase already includes scaffolding for LSTM and XGBoost models (exported to ONNX)
so you can drop your own predictors into `src/ml/` and call them from the
Back-tester or a new “Forecast” page without touching the UI plumbing.
"""
)
