"""Agent CRUD + retrieval query — HTTP layer only."""

from uuid import UUID

from fastapi import APIRouter, HTTPException

from database import get_supabase
from models.agent import (
    Agent,
    AgentCreate,
    AgentUpdate,
    AgentWithDocuments,
    QueryRequest,
    QueryResponse,
)
from models.document import Document
from services import rag

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("", response_model=Agent, status_code=201)
def create_agent(payload: AgentCreate) -> Agent:
    db = get_supabase()
    res = db.table("agents").insert(payload.model_dump()).execute()
    if not res.data:
        raise HTTPException(500, "failed to create agent")
    return Agent(**res.data[0])


@router.get("/{agent_id}", response_model=AgentWithDocuments)
def get_agent(agent_id: UUID) -> AgentWithDocuments:
    db = get_supabase()
    agent_res = db.table("agents").select("*").eq("id", str(agent_id)).maybe_single().execute()
    if not agent_res.data:
        raise HTTPException(404, "agent not found")

    docs_res = (
        db.table("documents")
        .select("*")
        .eq("agent_id", str(agent_id))
        .order("created_at", desc=True)
        .execute()
    )
    documents = [Document(**row) for row in (docs_res.data or [])]
    return AgentWithDocuments(**agent_res.data, documents=documents)


@router.patch("/{agent_id}", response_model=Agent)
def update_agent(agent_id: UUID, payload: AgentUpdate) -> Agent:
    db = get_supabase()
    updates = payload.model_dump(exclude_unset=True)
    if updates:
        db.table("agents").update(updates).eq("id", str(agent_id)).execute()
    res = db.table("agents").select("*").eq("id", str(agent_id)).maybe_single().execute()
    if not res.data:
        raise HTTPException(404, "agent not found")
    return Agent(**res.data)


@router.post("/{agent_id}/query", response_model=QueryResponse)
def query_agent(agent_id: UUID, payload: QueryRequest) -> QueryResponse:
    """Debug retrieval — returns top-k chunks for a query string."""
    return QueryResponse(matches=rag.retrieve(agent_id, payload.query))


@router.get("/{agent_id}/calls")
def list_calls(agent_id: UUID) -> list[dict]:
    raise HTTPException(status_code=501, detail="not implemented")
