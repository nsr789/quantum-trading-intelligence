from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import streamlit as st
from src.knowledge.vectorstore import load_vectorstore
from src.agents.crew_rag import qa_with_crew
from src.config.constants import DEFAULT_TICKERS

st.markdown("# 💬 Company Q&A (CrewAI + Chroma)")

ticker = st.selectbox("Ticker", DEFAULT_TICKERS, 0)
question = st.text_input("Ask a question about the company", "What does the company do?")
if st.button("Answer"):
    with st.spinner("Running Crew…"):
        answer = qa_with_crew(question, ticker)
    st.markdown(answer)

    # show source documents
    col = load_vectorstore()
    docs = col.query(query_texts=[question], n_results=3)
    st.expander("Top context snippets").write(docs["documents"][0])
