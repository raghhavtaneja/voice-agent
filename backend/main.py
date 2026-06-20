"""FastAPI entrypoint — wires routers, nothing else."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import settings
from routers import agents, documents, webhook
from services import rag

logging.basicConfig(level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm Pinecone's embedding model so the first caller doesn't eat the
    # ~2.6s cold start on their opening question.
    rag.warmup()
    yield


app = FastAPI(title="Voice Agent API", lifespan=lifespan)

app.include_router(agents.router)
app.include_router(documents.router)
app.include_router(webhook.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
