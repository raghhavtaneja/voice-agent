# AI Voice Agent with RAG

## What We're Building
A platform where businesses create AI voice agents backed by their own documents.
A business uploads PDFs (FAQs, manuals, policies), gets a real phone number, and
callers are answered by an AI that speaks only from those documents — no hallucination.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Voice pipeline | Vapi.ai (telephony + STT + TTS + turn-taking) |
| Backend | Python + FastAPI |
| LLM | Claude API (claude-sonnet-4-5) |
| PDF parsing | PyMuPDF (fitz) |
| Embeddings + vector store | Pinecone integrated index (llama-text-embed-v2, 1024 dims) |
| Database | PostgreSQL via Supabase |
| File storage | Supabase Storage |
| Frontend | Next.js + Tailwind CSS |
| Backend hosting | Railway (main branch only) |
| Frontend hosting | Vercel (main branch only) |

---

## Repo Structure

```
/
├── backend/
│   ├── main.py              # FastAPI app entry + router registration
│   ├── config.py            # All env vars via pydantic-settings
│   ├── database.py          # Supabase client singleton
│   ├── models/
│   │   ├── agent.py         # Agent Pydantic schemas
│   │   └── document.py      # Document Pydantic schemas
│   ├── routers/
│   │   ├── agents.py        # Agent CRUD endpoints
│   │   ├── documents.py     # PDF upload + processing endpoints
│   │   └── webhook.py       # Vapi webhook handler
│   ├── services/
│   │   ├── rag.py           # Embed + retrieve (Pinecone)
│   │   ├── llm.py           # Claude API calls
│   │   └── vapi.py          # Vapi API wrapper
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── (Next.js app)
├── CLAUDE.md
└── README.md
```

---

## Data Models

```
Agent
  id, name, voice_id, personality, guidelines, vapi_phone_number_id, created_at

Document
  id, agent_id, filename, file_url, status (processing | ready | failed), created_at

CallLog
  id, agent_id, vapi_call_id, transcript, duration_seconds, created_at
```

---

## How a Call Works (Runtime Flow)

```
Caller dials number
  → Vapi handles audio (STT)
  → Vapi sends transcript to POST /webhook/vapi (FastAPI)
  → We embed the caller's message
  → Query Pinecone for top-4 chunks filtered by agent_id
  → Build prompt: personality + guidelines + RAG chunks + conversation history
  → Send to Claude API
  → Return response text to Vapi
  → Vapi speaks it (TTS)
```

---

## RAG Config

- Chunking: 200 tokens, 40 token overlap, split on paragraphs first (blank lines, incl. whitespace-only)
- Embeddings: handled server-side by Pinecone integrated index (`llama-text-embed-v2`, 1024 dims)
- Pinecone namespace: `agent_{agent_id}` (isolates each agent's knowledge base)
- Top-k retrieval: 4 chunks per turn
- Per-record fields: `_id`, `chunk_text` (embedded), `agent_id`, `document_id`, `filename`, `page_number`

---

## API Endpoints

```
POST /agents                  → create agent
GET  /agents/{id}             → get agent + documents
PATCH /agents/{id}            → update agent
POST /agents/{id}/documents   → upload PDF
POST /webhook/vapi            → Vapi call turn handler
GET  /agents/{id}/calls       → call logs
```

---

## RAG Prompt Structure

```
You are {agent_name}. {personality}

Guidelines:
{guidelines}

Relevant information from our documents:
{retrieved_chunks}

Respond naturally and concisely. You are speaking out loud — no bullet points,
no markdown. Keep answers under 3 sentences unless the caller asks for more detail.
If the documents don't contain the answer, say so honestly and offer to connect
them with a human.
```

---

## Code Style Rules

- **Routers handle HTTP only** — no business logic, just call services and return responses
- **Services handle all logic** — RAG, LLM calls, Vapi API, PDF processing
- **config.py is the only place for env vars** — never use `os.getenv` elsewhere
- **No unused imports**
- **Pydantic models for all request/response shapes** — no raw dicts across boundaries
- Keep functions small and single-purpose

---

## Git Branching

- `main` — production, deployed to Railway + Vercel
- `dev` — integration branch, merge features here first
- `feature/*` — branch off dev, merge back to dev when done
- Never commit directly to main

---

Week 1 — RAG pipeline
- [x] FastAPI scaffold
- [x] PDF upload → parse → chunk → embed → upsert to Pinecone
- [x] Query endpoint: text in → top-k chunks out
- [x] Test retrieval with a real PDF

## Current Phase

Week 2 — Vapi Integration
- [ ] Agent CRUD endpoints
- [ ] Document upload endpoint (triggers RAG pipeline)
- [ ] Vapi webhook: receive turn → RAG → Claude → return response
- [ ] Provision Vapi phone number via API, link to agent
- [ ] End-to-end test call
