import os

from langchain.schema import HumanMessage
from src.utils.llm import chat_stream


def test_dummy_llm(monkeypatch):
    """Clears env to force DummyLLM path."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    chunks = list(
        chat_stream([HumanMessage(content="hello, world!")])
    )  # should not raise

    assert "LLM keys missing" in chunks[0]
    
