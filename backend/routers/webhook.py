"""Vapi webhook — OpenAI-compatible Custom LLM endpoint (HTTP layer only)."""

import time
from collections.abc import Iterator
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from database import get_supabase
from models.webhook import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)
from services import conversation

router = APIRouter(prefix="/webhook", tags=["webhook"])


@router.post("/vapi/{agent_id}/chat/completions")
def vapi_webhook(agent_id: UUID, payload: ChatCompletionRequest):
    # Vapi treats model.url as an OpenAI base URL and POSTs to
    # {url}/chat/completions, so the assistant's url is
    # https://<host>/webhook/vapi/<agent_id> and we route on the path.
    db = get_supabase()
    agent = (
        db.table("agents")
        .select("name, personality, guidelines")
        .eq("id", str(agent_id))
        .maybe_single()
        .execute()
    )
    if not agent.data:
        raise HTTPException(404, "agent not found")

    history = [
        {"role": m.role, "content": m.content}
        for m in payload.messages
        if m.role in ("user", "assistant") and m.content
    ]

    completion_id = f"chatcmpl-{uuid4().hex}"
    created = int(time.time())
    kwargs = {
        "name": agent.data["name"],
        "personality": agent.data["personality"],
        "guidelines": agent.data["guidelines"],
        "history": history,
    }

    # Vapi sets stream=true and reads the reply as an OpenAI SSE stream of
    # chat.completion.chunk events; a plain JSON body leaves it with nothing
    # to speak (call drops on silence). Our own tester sends stream=false.
    if payload.stream:
        deltas = conversation.generate_reply_stream(agent_id, **kwargs)
        return StreamingResponse(
            _stream_chunks(completion_id, created, deltas),
            media_type="text/event-stream",
        )

    reply = conversation.generate_reply(agent_id, **kwargs)
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=settings.groq_model,
        choices=[ChatCompletionChoice(message=ChatCompletionMessage(content=reply.text))],
        usage=Usage(
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.output_tokens,
            total_tokens=reply.total_tokens,
        ),
    )


def _stream_chunks(completion_id: str, created: int, deltas: Iterator[str]) -> Iterator[str]:
    """Relay Gemini's text deltas as OpenAI chat.completion.chunk SSE events."""

    def sse(choice: ChatCompletionChunkChoice) -> str:
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=settings.groq_model,
            choices=[choice],
        )
        return f"data: {chunk.model_dump_json()}\n\n"

    first = True
    for delta in deltas:
        # OpenAI convention: the first chunk carries the role, the rest content.
        role = "assistant" if first else None
        first = False
        yield sse(ChatCompletionChunkChoice(delta=ChatCompletionDelta(role=role, content=delta)))
    yield sse(ChatCompletionChunkChoice(delta=ChatCompletionDelta(), finish_reason="stop"))
    yield "data: [DONE]\n\n"
