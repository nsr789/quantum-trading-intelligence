from __future__ import annotations

import os
from typing import Callable

# ▼ correct import
from crewai import Agent, Crew, Task
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

from src.utils.llm import chat_stream
from src.knowledge.vectorstore import load_vectorstore

# ──────────────────────────── LLM selector ────────────────────────────────
def _dummy_llm(prompt: str) -> str:
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


# ────────────────────────── Crew-powered QA  ──────────────────────────────
def qa_with_crew(question: str, ticker: str) -> str:
    """Answer `question` about `ticker` via CrewAI + Chroma or offline fallback."""
    col = load_vectorstore()
    docs = col.query(query_texts=[question], n_results=3)
    context = "\n".join(docs["documents"][0])

    llm_backend = _select_llm()

    # Offline deterministic answer keeps tests network-free
    if llm_backend is _dummy_llm:
        snippet = docs["documents"][0][0]
        return (
            f"**Heuristic answer for {ticker} (offline mode)**\n\n"
            f"Top context snippet:\n> {snippet}"
        )

    # Build agents & tasks
    researcher = Agent(
        name="Researcher",
        role="Data collector",
        goal="Provide relevant context snippets",
        backstory="Access to a vector DB of company knowledge.",
        allow_delegation=False,
    )
    analyst = Agent(
        name="Analyst",
        role="Equity analyst",
        goal="Answer user questions concisely",
        backstory="Uses collected context and market knowledge.",
        allow_delegation=False,
    )

    tasks = [
        Task(
            description=(
                f"Retrieve key facts about {ticker} needed to answer:\n'{question}'. "
                "Use only the provided context.\n\n"
                f"Context:\n{context}"
            ),
            expected_output="Bullet list of relevant facts",
            agent=researcher,
        ),
        Task(
            description="Craft a two-paragraph markdown answer using the facts.",
            expected_output="Markdown answer",
            agent=analyst,
        ),
    ]

    crew = Crew(agents=[researcher, analyst], tasks=tasks, llm=llm_backend)
    result = crew.kickoff()

    # CrewAI ≥0.4 returns CrewOutput(raw=...)
    if hasattr(result, "raw"):
        return str(result.raw).strip()

    return str(result).strip()
