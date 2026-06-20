# AI Voice Agent with RAG

Businesses upload PDFs, get a phone number, and callers reach an AI that answers
only from those documents. See `CLAUDE.md` for the full spec.

## Infra prerequisites (one-time)

1. **Supabase** project. Run `backend/sql/schema.sql` in the SQL editor. Create a
   public Storage bucket named `documents` (or change `SUPABASE_STORAGE_BUCKET`).
2. **Pinecone** account — copy your API key into `.env`. The integrated index
   (bound to `llama-text-embed-v2`) gets created by the setup script below.

## Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in keys
python -m scripts.setup_pinecone  # creates the Pinecone integrated index
uvicorn main:app --reload
```

Health check: `curl http://localhost:8000/health`

## Week 1 smoke test

```bash
# 1) create an agent
curl -s -X POST http://localhost:8000/agents \
  -H 'content-type: application/json' \
  -d '{
    "name": "Acme Support",
    "voice_id": "vapi-default",
    "personality": "friendly, concise",
    "guidelines": "be honest when you do not know"
  }'
# → returns {"id": "<AGENT_ID>", ...}

# 2) upload a PDF
curl -s -X POST http://localhost:8000/agents/<AGENT_ID>/documents \
  -F file=@/path/to/handbook.pdf

# 3) test retrieval
curl -s -X POST http://localhost:8000/agents/<AGENT_ID>/query \
  -H 'content-type: application/json' \
  -d '{"query": "what is the refund policy?"}'
# → returns top-4 chunks with filename + page_number + score
```

## Layout

- `backend/routers/` — HTTP only, no business logic
- `backend/services/` — RAG, LLM, Vapi
- `backend/models/` — Pydantic shapes
- `backend/config.py` — env vars (only place `os.getenv` lives)
- `backend/sql/schema.sql` — Supabase tables
- `backend/scripts/setup_pinecone.py` — one-time Pinecone index creation
