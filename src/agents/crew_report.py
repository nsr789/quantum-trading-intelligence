# ─────────────────────────── src/agents/crew_report.py ──────────────────────
from __future__ import annotations

import os
from typing import Callable, Dict

import yfinance as yf
from crewai import Agent, Crew, Task
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

from src.data.cache import cached
from src.data.macro import get_unemployment_rate
from src.agents.tools import company_news_sentiment         # <- NEW
from src.utils.llm   import chat_stream
from src.utils.logging import get_logger

log = get_logger(module="crew_report")

# ───────────────────────────── LLM helper ────────────────────────────────────
def _dummy_llm(prompt: str) -> str:                       # offline fallback
    return "".join(chat_stream([HumanMessage(content=prompt)]))


def _select_llm() -> Callable | ChatOpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return ChatOpenAI(api_key=key,
                          model_name="gpt-3.5-turbo-0125",
                          temperature=0)
    return _dummy_llm


# ───────────────────── fundamentals helper (unchanged) ──────────────────────
def fundamentals(ticker: str) -> Dict[str, str | float]:
    try:
        info = yf.Ticker(ticker).info
        return {
            "ttm_pe":     info.get("trailingPE"),
            "rev_growth": info.get("revenueGrowth"),
            "sector":     info.get("sector"),
        }
    except Exception as exc:                                 # noqa: BLE001
        log.warning("fundamentals_fetch_failed", error=str(exc))
        return {}


# ──────────────────────────── main entry point ──────────────────────────────
@cached           # joblib-cache so repeat tickers cost ≈0 tokens
def generate_report(ticker: str) -> str:
    """
    5-bullet equity research note via CrewAI.

    * News sentiment: FinBERT on full articles → TextBlob fallback.
    * Macro: latest unemployment.
    * Valuation: lite fundamentals from yfinance.
    """
    # ---------- SENTIMENT ---------------------------------------------------
    try:
        sent = company_news_sentiment(ticker, limit=30, full_article=True)
    except Exception as exc:                                # noqa: BLE001
        log.warning("finbert_failed_fallback_textblob", error=str(exc))
        sent = company_news_sentiment(ticker, limit=30, full_article=False)

    # ---------- MACRO -------------------------------------------------------
    macro = {"unemployment": get_unemployment_rate()}

    # ---------- FUNDAMENTALS -----------------------------------------------
    facts = fundamentals(ticker)

    # ---------- OFFLINE FALLBACK (no API keys) ------------------------------
    llm = _select_llm()
    if llm is _dummy_llm:
        bias = max(sent, key=sent.get)
        pos, neg, neu = [s * 100 for s in sent.values()]
        return (
            f"- Overall news sentiment for **{ticker}** is *{bias}* "
            f"({pos:.0f}% ↑, {neg:.0f}% ↓, {neu:.0f}% →)\n"
            "- Market tone remains driven by headline momentum\n"
            "- Watch EPS revisions versus sentiment trend for confirmation\n"
            "- Positive tone may support near-term price action\n"
            "- Re-evaluate if negative share of headlines exceeds 30 %"
        )

    # ---------- CrewAI agents ----------------------------------------------
    sentiment_agent = Agent(
        name="Sentiment Analyst",
        role="News-Sentiment Analyst",
        goal="Summarise headline sentiment",
        backstory="Quantifies positive/negative tone from recent news.",
        allow_delegation=False,
    )
    macro_agent = Agent(
        name="Macro Analyst",
        role="Macro Analyst",
        goal="Comment on macro backdrop",
        backstory="Uses unemployment & CPI data.",
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
            description=f"Provide one sentence on news sentiment: {sent}",
            expected_output="Sentiment bullet",
            agent=sentiment_agent,
        ),
        Task(
            description=(
                f"Provide one sentence on macro backdrop "
                f"(unemployment_rate={macro['unemployment']})"
            ),
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
                "Combine the three analyst bullets into **exactly five** "
                f"markdown bullet points that give market colour on {ticker}."
            ),
            expected_output="5 markdown bullets",
            agent=editor,
        ),
    ]

    crew   = Crew([sentiment_agent, macro_agent, val_agent, editor],
                  tasks, llm)
    result = crew.kickoff()
    return str(getattr(result, "raw", result)).strip()
