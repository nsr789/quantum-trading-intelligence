from __future__ import annotations

import os
from typing import Generator, List

from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from src.utils.logging import get_logger

log = get_logger(module="llm")

try:
    from langchain_community.chat_models import ChatGroq
except ImportError:  # pragma: no cover
    ChatGroq = None  # type: ignore

try:
    from langchain_openai import ChatOpenAI
    from openai.types.chat import ChatCompletionChunk
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore
    ChatCompletionChunk = None  # type: ignore


class DummyLLM:
    """Offline fallback – returns a single explanatory reply."""

    def stream(self, msgs: List[BaseMessage]):  # noqa: D401
        last = msgs[-1].content if msgs else ""
        yield (
            "LLM keys missing – using DummyLLM fallback.\n\n"
            "Echo: " + last
        )


def _select_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key and ChatGroq is not None:
        return ChatGroq(
            api_key=groq_key,
            model_name="mixtral-8x7b-32768",
            temperature=0,
            streaming=True,
            max_tokens=1024,
        )

    if openai_key and ChatOpenAI is not None:
        return ChatOpenAI(
            api_key=openai_key,
            model_name="gpt-3.5-turbo-0125",
            temperature=0,
            streaming=True,
            max_tokens=1024,
        )

    return DummyLLM()


def _chunk_to_text(chunk) -> str:
    """Return plain text from an OpenAI or Groq stream chunk."""
    # OpenAI python-sdk
    if ChatCompletionChunk and isinstance(chunk, ChatCompletionChunk):
        return chunk.choices[0].delta.content or ""
    # Groq SDK returns dict-like objects inside LangChain
    if hasattr(chunk, "choices") and hasattr(chunk.choices[0], "message"):
        return chunk.choices[0].message.content or ""
    # Fallback – stringify
    return str(chunk)


def chat_stream(
    messages: List[BaseMessage] | List[dict[str, str]]
) -> Generator[str, None, None]:
    """
    Yield text tokens from the chosen LLM.

    On *any* exception (rate-limit, network, auth) auto-fallback to DummyLLM.
    """
    if messages and isinstance(messages[0], dict):
        mapping = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        messages = [mapping[m["role"]](content=m["content"]) for m in messages]  # type: ignore

    llm = _select_llm()

    try:
        for chunk in llm.stream(messages):  # type: ignore[attr-defined]
            text = _chunk_to_text(chunk)
            if text:
                yield text
    except Exception as exc:  # noqa: BLE001
        log.warning(f"LLM backend failed – falling back to DummyLLM: {exc}")
        for chunk in DummyLLM().stream(messages):
            yield chunk
