"""Pydantic shapes for Agent requests/responses."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from models.document import Document


class AgentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    voice_id: str
    personality: str
    guidelines: str


class AgentUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    voice_id: str | None = None
    personality: str | None = None
    guidelines: str | None = None


class Agent(BaseModel):
    id: UUID
    name: str
    voice_id: str
    personality: str
    guidelines: str
    vapi_phone_number_id: str | None = None
    created_at: datetime


class AgentWithDocuments(Agent):
    documents: list[Document] = []


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class QueryMatch(BaseModel):
    text: str
    filename: str
    page_number: int
    score: float


class QueryResponse(BaseModel):
    matches: list[QueryMatch]
