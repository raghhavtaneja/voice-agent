"""Pydantic shapes for Document requests/responses."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel


class DocumentStatus(str, Enum):
    processing = "processing"
    ready = "ready"
    failed = "failed"


class Document(BaseModel):
    id: UUID
    agent_id: UUID
    filename: str
    file_url: str
    status: DocumentStatus
    created_at: datetime
