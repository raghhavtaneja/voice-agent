"""LLM wrapper — builds the RAG prompt and returns spoken text.

CLAUDE.md targets Claude (claude-sonnet-4-5) for production. We're temporarily
on Groq's free tier (Llama 3.3) — generous quota and the fastest inference,
which also helps call latency. Groq is OpenAI-compatible, so the system prompt
is just the first message and assistant turns use role ``assistant``. To switch
to Claude/Gemini later, only ``respond``/``respond_stream`` change.
"""

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

from groq import Groq, InternalServerError

from config import settings

logger = logging.getLogger("voice_agent.llm")

# Groq can intermittently return 5xx under load. Retry a few times with backoff
# so a transient blip doesn't become dead air on a call.
_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 0.8
_MAX_OUTPUT_TOKENS = 512


SYSTEM_TEMPLATE = """You are {agent_name}. {personality}

Guidelines:
{guidelines}

Relevant information from our documents:
{retrieved_chunks}

Respond naturally and concisely. You are speaking out loud — no bullet points,
no markdown. Keep answers under 3 sentences unless the caller asks for more detail.
Always reply in the same language the caller used (e.g. answer in Hindi if they
spoke Hindi), even though the documents above may be in another language.
If the documents don't contain the answer, say so honestly and offer to connect
them with a human."""


@lru_cache
def _client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


def build_system_prompt(
    agent_name: str,
    personality: str,
    guidelines: str,
    retrieved_chunks: list[str],
) -> str:
    return SYSTEM_TEMPLATE.format(
        agent_name=agent_name,
        personality=personality,
        guidelines=guidelines,
        retrieved_chunks="\n\n".join(retrieved_chunks) or "(no relevant context found)",
    )


@dataclass
class Reply:
    """The LLM's text answer plus the token counts for the call."""

    text: str
    prompt_tokens: int
    output_tokens: int
    total_tokens: int


# Running totals for the life of the process — handy for watching usage during
# a test session.
_session_total_tokens = 0
_session_calls = 0


def _build_messages(system_prompt: str, history: list[dict]) -> list[dict]:
    # OpenAI/Groq shape: system prompt is the first message; assistant turns
    # keep role ``assistant`` (unlike Gemini's ``model``).
    return [{"role": "system", "content": system_prompt}, *history]


def respond(system_prompt: str, history: list[dict]) -> Reply:
    """Send conversation to Groq and return the full reply plus token usage.

    Non-streaming — used by the debug tester (scripts/ask.py). The live webhook
    uses ``respond_stream`` to cut time-to-first-word.
    """
    messages = _build_messages(system_prompt, history)
    for attempt in range(_MAX_RETRIES):
        try:
            resp = _client().chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                max_tokens=_MAX_OUTPUT_TOKENS,
            )
            text = resp.choices[0].message.content or ""
            return _record_usage(text, resp.usage)
        except InternalServerError:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
    return Reply("", 0, 0, 0)  # unreachable; loop either returns or raises


def respond_stream(system_prompt: str, history: list[dict]) -> Iterator[str]:
    """Yield Groq's reply as text deltas as they're generated (low latency).

    Token usage rides on the final chunk (``include_usage``) and is logged when
    the stream completes. Retries only apply before the first token is emitted —
    once we've yielded text we can't safely restart without duplicating it.
    """
    messages = _build_messages(system_prompt, history)
    for attempt in range(_MAX_RETRIES):
        emitted = False
        usage = None
        try:
            stream = _client().chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                max_tokens=_MAX_OUTPUT_TOKENS,
                stream=True,
            )
            for chunk in stream:
                # Groq attaches final token usage to the last chunk's x_groq.
                if (xg := getattr(chunk, "x_groq", None)) and xg.usage is not None:
                    usage = xg.usage
                if chunk.choices and (delta := chunk.choices[0].delta.content):
                    emitted = True
                    yield delta
            _record_usage("", usage)
            return
        except InternalServerError:
            if emitted or attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_RETRY_BACKOFF_S * (attempt + 1))


def _record_usage(text: str, usage) -> Reply:
    """Log per-call token usage and a running session total; return a Reply."""
    global _session_total_tokens, _session_calls
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    output = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or 0

    _session_total_tokens += total
    _session_calls += 1
    logger.info(
        "groq tokens: prompt=%d output=%d total=%d | session: %d tokens / %d calls",
        prompt,
        output,
        total,
        _session_total_tokens,
        _session_calls,
    )
    return Reply(
        text=text,
        prompt_tokens=prompt,
        output_tokens=output,
        total_tokens=total,
    )
