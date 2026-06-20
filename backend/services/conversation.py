"""One call turn: conversation history -> RAG -> Claude -> reply text.

Stateless — Vapi owns turn-taking and replays the full history each turn, so
we just ground the latest question in the agent's documents and answer.
"""

from collections.abc import Iterator
from uuid import UUID

from services import llm, rag


def _prepare(
    agent_id: UUID,
    *,
    name: str,
    personality: str,
    guidelines: str,
    history: list[dict],
) -> tuple[str, list[dict]]:
    """Ground the latest user turn in the agent's docs -> (system_prompt, messages)."""
    query = _latest_user_message(history)
    chunks = rag.retrieve(agent_id, query) if query else []

    system_prompt = llm.build_system_prompt(
        agent_name=name,
        personality=personality,
        guidelines=guidelines,
        retrieved_chunks=[c["text"] for c in chunks],
    )

    # No user turn yet (e.g. opening the call) — seed a greeting so the model
    # has a non-empty user message to respond to.
    messages = history or [{"role": "user", "content": "Hello"}]
    return system_prompt, messages


def generate_reply(
    agent_id: UUID,
    *,
    name: str,
    personality: str,
    guidelines: str,
    history: list[dict],
) -> llm.Reply:
    """Full (non-streaming) reply + token usage — used by the debug tester."""
    system_prompt, messages = _prepare(
        agent_id, name=name, personality=personality, guidelines=guidelines, history=history
    )
    return llm.respond(system_prompt, messages)


def generate_reply_stream(
    agent_id: UUID,
    *,
    name: str,
    personality: str,
    guidelines: str,
    history: list[dict],
) -> Iterator[str]:
    """Stream the reply as text deltas — used by the live Vapi webhook."""
    system_prompt, messages = _prepare(
        agent_id, name=name, personality=personality, guidelines=guidelines, history=history
    )
    return llm.respond_stream(system_prompt, messages)


def _latest_user_message(history: list[dict]) -> str:
    for msg in reversed(history):
        if msg["role"] == "user":
            return msg["content"]
    return ""
