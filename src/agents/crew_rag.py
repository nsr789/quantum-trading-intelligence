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
def qa_with_crew(question: str, ticker: str) -> str:
    """
    Answer *question* about *ticker*:
    1. ensure ticker docs present in Chroma (yfinance bootstrap)
    2. embed-search with ticker filter → top-k
    3. simple semantic re-ranking
    4. 2-agent CrewAI answer
    5. offline deterministic fallback for pytest / no-key env
    """
    col = load_vectorstore(ticker)
    res = col.query(query_texts=[question], n_results=8, where={"ticker": ticker})
    docs = res["documents"][0] or ["(no matching documents)"]
    docs = _rerank(question, docs)[:3]  # keep best 3 after re-rank
    context = "\n".join(f"- {d}" for d in docs)

    llm_backend = _select_llm()

    # offline deterministic answer (for CI / tests)
    if llm_backend is _dummy_llm:
        return f"**Heuristic answer for {ticker} (offline mode)**\n\nContext snippet:\n> {docs[0]}"

    # ────────────── build agents ────────────────────────────────────────────
    system_prompt = SystemMessage(
        content=(
            "You are a sell-side equity analyst. "
            "Answer the user strictly with the supplied context. "
            "If insufficient information is present, say 'Not enough data.'"
        )
    )

    researcher = Agent(
        name="Researcher",
        role="Context Retriever",
        goal="Summarise the key facts from the provided context.",
        backstory="Has access to a company knowledge vector DB.",
        allow_delegation=False,
    )
    analyst = Agent(
        name="Analyst",
        role="Equity Analyst",
        goal="Provide a concise, accurate answer.",
        backstory="Uses the summarised facts to craft the final response.",
        allow_delegation=False,
    )

    tasks = [
        Task(
            description=f"From the context below extract the facts needed to answer:\n\n'{question}'\n\nContext:\n{context}",
            expected_output="Bullet list of facts",
            agent=researcher,
        ),
        Task(
            description="Write a two-paragraph markdown answer based **only** on the facts.",
            expected_output="Final answer",
            agent=analyst,
        ),
    ]

    crew = Crew(agents=[researcher, analyst], tasks=tasks, llm=llm_backend, messages=[system_prompt])
    result = crew.kickoff()

    # CrewAI ≥0.4 returns CrewOutput(raw=...)
    return str(getattr(result, "raw", result)).strip()
