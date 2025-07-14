from __future__ import annotations

import os  # ← added
from langchain.schema import HumanMessage, SystemMessage

from src.utils.llm import chat_stream
from src.agents.tools import company_news_sentiment

SYS_PROMPT = """
You are an equity research assistant. Respond with concise bullet-points.
"""


def _heuristic_report(ticker: str, sent: dict[str, float]) -> str:
    """Return a deterministic 5-bullet note without calling an LLM."""
    pos, neg, neu = sent["positive"], sent["negative"], sent["neutral"]
    bias = "positive" if pos > neg else "negative" if neg > pos else "neutral"
    pos, neg, neu = pos * 100, neg * 100, neu * 100
    return (
        f"- Overall news sentiment for **{ticker}** is *{bias}* "
        f"({pos:.0f}% ↑, {neg:.0f}% ↓, {neu:.0f}% →)\n"
        "- Market tone remains driven by headline momentum\n"
        "- Watch EPS revisions versus sentiment trend for confirmation\n"
        "- Positive tone may support near-term price action\n"
        "- Re-evaluate if negative share of headlines exceeds 30 %"
    )


def _needs_fallback(txt: str) -> bool:
    """Detect DummyLLM output prefixes."""
    return txt.startswith("⚠️") or txt.startswith("LLM keys missing")


def generate_report(ticker: str) -> str:
    """Generate a 5-bullet markdown report.

    * Uses Groq ➔ OpenAI if valid keys/quota exist.
    * Otherwise returns a heuristic report.
    """
    sent = company_news_sentiment(ticker, limit=20)

    # Fast path – no keys at all
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")):
        return _heuristic_report(ticker, sent)

    prompt = (
        f"Write a 5-bullet market colour on {ticker}. "
        f"News sentiment: {sent}. Use markdown bullet points."
    )

    chunks = chat_stream(
        [SystemMessage(content=SYS_PROMPT), HumanMessage(content=prompt)]
    )
    text = "".join(chunks).strip()

    if _needs_fallback(text):
        return _heuristic_report(ticker, sent)

    return text
