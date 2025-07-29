from __future__ import annotations

import os
from importlib import reload
from typing import Callable

from crewai import Agent, Crew, Task
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.utils.llm import chat_stream
from src.knowledge.vectorstore import load_vectorstore

# ─────────────────── LLM selector (OpenAI when key present) ──────────────────
def _dummy_llm(prompt: str) -> str:
    return "".join(chat_stream([HumanMessage(content=prompt)]))


def _select_llm() -> Callable | ChatOpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return ChatOpenAI(api_key=key, model_name="gpt-3.5-turbo-0125", temperature=0)
    return _dummy_llm


# ──────────────────────────── helper: re-rank top-k docs ─────────────────────
def _rerank(question: str, docs: list[str]) -> list[str]:
    """Very light cosine re-rank using the same embedding function (CPU-cheap)."""
    from sentence_transformers import util

    from src.knowledge.vectorstore import EMBED  # reuse singleton

    q_emb = EMBED._embed_fn([question])[0]  # type: ignore (private attr)
    d_emb = EMBED._embed_fn(docs)           # type: ignore

    sims = util.cos_sim(q_emb, d_emb).tolist()[0]
    return [d for _, d in sorted(zip(sims, docs), reverse=True)]


# ────────────────────────── Crew-powered Q-A function ────────────────────────
# ───────── Crew-powered Q-A ─────────
def qa_with_crew(question: str, ticker: str) -> str:
    col   = load_vectorstore(ticker)
    raw   = col.query(query_texts=[question],
                      n_results=12,           # <- more recall
                      where={"ticker": ticker})
    docs  = raw["documents"][0] or ["(no context)"]
    docs  = _rerank(question, docs)[:3]
    ctx   = "\n".join(f"{i+1}. {d}" for i, d in enumerate(docs))

    llm   = _select_llm()
    if llm is _dummy_llm:
        return f"**Heuristic answer for {ticker}**\n\n> {docs[0]}"

    researcher = Agent(
        name="Researcher", role="Context retriever",
        goal="Extract only the sentences directly relevant to the question.",
        backstory="Has access to a vector DB seeded with company fact-sheets.",
        allow_delegation=False,
    )
    analyst = Agent(
        name="Analyst", role="Equity analyst",
        goal="Answer the user’s question as accurately as possible.",
        backstory="Uses the researcher’s citations; no fabrication allowed.",
        allow_delegation=False,
    )

    tasks = [
        Task(
            description=(
                f"Given the numbered context snippets below, pick the lines "
                f"that answer **exactly** this question: '{question}'. "
                f"Return them as bullet points *with the line number*.\n\n"
                f"Context:\n{ctx}"
            ),
            expected_output="Bullet list like '- [2] Apple…'",
            agent=researcher,
        ),
        Task(
            description=(
                "Write a concise, two-paragraph markdown answer **using only "
                "those bullet-point facts**. Do not add external knowledge."
            ),
            expected_output="Markdown answer",
            agent=analyst,
        ),
    ]

    crew = Crew([researcher, analyst], tasks, llm)
    result = crew.kickoff()
    return str(getattr(result, "raw", result)).strip()
