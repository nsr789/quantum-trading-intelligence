# ───────────────────────── src/agents/crew_report.py ─────────────────────────
from __future__ import annotations

import os
from typing import Callable, Dict, List

import pandas as pd
import yfinance as yf
from crewai import Agent, Crew, Task
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from textblob import TextBlob

from src.agents.sentiment import transformer_sentiment         # NEW
from src.data.cache   import cached
from src.data.macro   import get_unemployment_rate
from src.data.news    import fetch_news
from src.utils.llm    import chat_stream
from src.utils.logging import get_logger

log = get_logger(module="crew_report")

# ───────────────────────────── LLM helpers ───────────────────────────────────
def _dummy_llm(prompt: str) -> str:  # offline fallback
    return "".join(chat_stream([HumanMessage(content=prompt)]))


def _select_llm() -> Callable | ChatOpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return ChatOpenAI(
            api_key=key,
            model_name="gpt-3.5-turbo-0125",
            temperature=0,
        )
    return _dummy_llm


# ───────────────────────── fundamentals helper ───────────────────────────────
def fundamentals(ticker: str) -> Dict[str, str | float]:
    """Return lite fundamentals dict; swallow network/ratelimit errors."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "ttm_pe":      info.get("trailingPE"),
            "rev_growth":  info.get("revenueGrowth"),
            "sector":      info.get("sector"),
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("fundamentals_fetch_failed", error=str(exc))
        return {}


# ───────────────────────── sentiment helper (FinBERT) ────────────────────────
def _headline_sentiment(headlines: List[str]) -> Dict[str, float]:
    """
    Classify `headlines` with FinBERT → proportional share.

    Falls back to TextBlob polarity if transformer fails (e.g. cold start).
    """
    try:
        labels = transformer_sentiment(headlines)            # ProsusAI/FinBERT
        share  = (
            pd.Series(labels)
            .value_counts(normalize=True)
            .reindex(["positive", "negative", "neutral"])
            .fillna(0.0)
        )
        return {k: round(v, 2) for k, v in share.items()}
    except Exception as exc:  # transformer load / CUDA OOM / etc.
        log.warning("FinBERT_failed_fallback_TextBlob", error=str(exc))
        polar = pd.Series(headlines).map(
            lambda t: TextBlob(str(t)).sentiment.polarity
        )
        pos = (polar >  0.10).mean()
        neg = (polar < -0.10).mean()
        neu = 1 - pos - neg
        return {"positive": round(pos, 2), "negative": round(neg, 2), "neutral": round(neu, 2)}


# ─────────────────────────────── main entry ──────────────────────────────────
@cached  # joblib cache → repeat calls are free
def generate_report(ticker: str) -> str:
    """Return a 5-bullet markdown research note via CrewAI or heuristic fallback."""
    import pandas as pd  # local import to avoid circular when cached

    # ---------- NEWS SENTIMENT (FinBERT) ------------------------------------
    news_df = fetch_news(ticker, limit=30)
    if news_df.empty:
        sentiment = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
    else:
        sentiment = _headline_sentiment(news_df["title"].tolist())

    # ---------- MACRO --------------------------------------------------------
    macro = {"unemployment": get_unemployment_rate()}

    # ---------- FUNDAMENTALS -------------------------------------------------
    facts = fundamentals(ticker)

    # ---------- FALLBACK if no LLM ------------------------------------------
    llm = _select_llm()
    if llm is _dummy_llm:
        bias = max(sentiment, key=sentiment.get)
        pos, neg, neu = [s * 100 for s in sentiment.values()]
        return (
            f"- Overall news sentiment for **{ticker}** is *{bias}* "
            f"({pos:.0f}% ↑, {neg:.0f}% ↓, {neu:.0f}% →)\n"
            "- Market tone remains driven by headline momentum\n"
            "- Watch EPS revisions versus sentiment trend for confirmation\n"
            "- Positive tone may support near-term price action\n"
            "- Re-evaluate if negative share of headlines exceeds 30 %"
        )

    # ---------- CrewAI agents -----------------------------------------------
    sentiment_agent = Agent(
        name="Sentiment Analyst",
        role="News-Sentiment Analyst",
        goal="Summarise headline sentiment",
        backstory="Quantifies positive/negative tone from news.",
        allow_delegation=False,
    )
    macro_agent = Agent(
        name="Macro Analyst",
        role="Macro Analyst",
        goal="Comment on macro backdrop",
        backstory="Access to unemployment & CPI data.",
        allow_delegation=False,
    )
    val_agent = Agent(
        name="Valuation Analyst",
        role="Valuation Analyst",
        goal="Highlight fundamental valuation metrics",
        backstory="Uses P/E and revenue growth from filings.",
        allow_delegation=False,
    )
    editor = Agent(
        name="Editor",
        role="Senior Editor",
        goal="Merge analyses into 5 concise bullets",
        backstory="Experienced equity-research editor.",
        allow_delegation=False,
    )

    tasks = [
        Task(
            description=f"Provide one sentence on news sentiment: {sentiment}",
            expected_output="Sentiment bullet",
            agent=sentiment_agent,
        ),
        Task(
            description=f"Provide one sentence on macro backdrop (unemployment={macro['unemployment']})",
            expected_output="Macro bullet",
            agent=macro_agent,
        ),
        Task(
            description=f"Provide one sentence on valuation metrics: {facts}",
            expected_output="Valuation bullet",
            agent=val_agent,
        ),
        Task(
            description=(
                "Combine the three analyst bullets into **exactly five** markdown "
                f"bullet points that give market colour on {ticker}."
            ),
            expected_output="5 markdown bullets",
            agent=editor,
        ),
    ]

    # NEW: use keyword arguments (crewai ≥0.4)
    crew = Crew(
        agents=[sentiment_agent, macro_agent, val_agent, editor],
        tasks=tasks,
        llm=llm,
    )

    result = crew.kickoff()
    return str(getattr(result, "raw", result)).strip()
# ──────────────────────────────────────────────────────────────────────────────
