"""RAG pipeline: parse PDF → chunk → upsert to Pinecone → retrieve.

Pinecone's integrated index embeds text server-side using ``llama-text-embed-v2``.
Namespace per agent: ``agent_{agent_id}``.
Per-record fields: ``_id``, ``text`` (the embedded field), ``agent_id``,
``document_id``, ``filename``, ``page_number``.
"""

import re
from functools import lru_cache
from uuid import UUID

import fitz  # PyMuPDF
import tiktoken
from pinecone import Pinecone

from config import settings

# cl100k_base is a fine proxy for budgeting chunk size; the actual embedding
# tokenizer lives server-side at Pinecone.
_ENC = tiktoken.get_encoding("cl100k_base")

_UPSERT_BATCH = 96  # Pinecone integrated upsert_records soft limit


@lru_cache
def _pc() -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


@lru_cache
def _index():
    return _pc().Index(settings.pinecone_index)


def _namespace(agent_id: UUID) -> str:
    return f"agent_{agent_id}"


def warmup() -> None:
    """Prime Pinecone's server-side embedding model at startup.

    The first query of a session pays a ~2.6s cold start while the embedding
    model spins up; a throwaway search on boot moves that cost off the first
    caller. Best-effort — never block startup if Pinecone is unreachable.
    """
    try:
        _index().search(namespace="_warmup", top_k=1, inputs={"text": "warmup"})
    except Exception:  # noqa: BLE001 — warmup must never crash boot
        pass


# ---------- PDF ----------


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return (page_number, text) for each non-empty page. 1-indexed pages."""
    pages: list[tuple[int, str]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for idx, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((idx, text))
    return pages


# ---------- Chunking ----------


def chunk_text(text: str) -> list[str]:
    """Paragraph-first chunking with a token window.

    Splits on blank lines, hard-splits any paragraph longer than ``chunk_tokens``,
    then greedily packs pieces into chunks of up to ``chunk_tokens`` with a
    ``chunk_overlap_tokens`` tail carried into the next chunk.
    """
    max_t = settings.chunk_tokens
    overlap_t = settings.chunk_overlap_tokens

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split on blank lines, treating whitespace-only lines as separators too —
    # PDF extraction often emits "\n \n" rather than a clean "\n\n", so a literal
    # split would find no paragraph boundaries and collapse a whole page into one
    # chunk.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sep = _ENC.encode("\n\n")

    # 1) Tokenize and hard-split any oversized paragraph.
    pieces: list[list[int]] = []
    for para in paragraphs:
        toks = _ENC.encode(para)
        if len(toks) <= max_t:
            pieces.append(toks)
            continue
        step = max(1, max_t - overlap_t)
        for i in range(0, len(toks), step):
            window = toks[i : i + max_t]
            if window:
                pieces.append(window)
            if i + max_t >= len(toks):
                break

    # 2) Greedily pack pieces, carrying an overlap tail between chunks.
    #    Invariant: every emitted chunk is <= max_t tokens. When a hard-split
    #    piece is already near max_t the tail won't fit — start fresh in that
    #    case (the windowing has already baked in overlap on those pieces).
    chunks: list[str] = []
    buf: list[int] = []
    for piece in pieces:
        addl = (len(sep) if buf else 0) + len(piece)
        if buf and len(buf) + addl > max_t:
            chunks.append(_ENC.decode(buf))
            tail = buf[-overlap_t:] if overlap_t else []
            if tail and len(tail) + len(sep) + len(piece) <= max_t:
                buf = tail + sep + piece
            else:
                buf = list(piece)
        else:
            if buf:
                buf += sep
            buf += piece
    if buf:
        chunks.append(_ENC.decode(buf))

    return chunks


# ---------- Pinecone ----------


def upsert_chunks(
    agent_id: UUID,
    document_id: UUID,
    filename: str,
    chunks: list[tuple[int, str]],
) -> None:
    """Upsert chunks to Pinecone; the index embeds ``text`` server-side.

    ``chunks`` is a list of ``(page_number, text)``.
    """
    if not chunks:
        return

    records = [
        {
            "_id": f"{document_id}:{i}",
            "text": text,
            "agent_id": str(agent_id),
            "document_id": str(document_id),
            "filename": filename,
            "page_number": page_num,
        }
        for i, (page_num, text) in enumerate(chunks)
    ]

    ns = _namespace(agent_id)
    idx = _index()
    for i in range(0, len(records), _UPSERT_BATCH):
        idx.upsert_records(namespace=ns, records=records[i : i + _UPSERT_BATCH])


def delete_document(agent_id: UUID, document_id: UUID) -> None:
    """Drop all chunks for a document from Pinecone.

    Uses list-by-prefix + delete-by-ID because filter-based delete isn't
    supported on serverless indexes. IDs are ``{document_id}:{i}``.
    """
    ns = _namespace(agent_id)
    idx = _index()
    for page in idx.list(prefix=f"{document_id}:", namespace=ns):
        ids = [v.id for v in page.vectors]
        if ids:
            idx.delete(ids=ids, namespace=ns)


def retrieve(agent_id: UUID, query: str) -> list[dict]:
    """Return top-k chunks above ``min_relevance_score`` in the agent's namespace.

    Pinecone always returns ``top_k`` matches even when none are good; the floor
    is what turns a weak match list into an honest empty list so the LLM is told
    the documents don't cover the question.
    """
    res = _index().search(
        namespace=_namespace(agent_id),
        top_k=settings.top_k,
        inputs={"text": query},
    )
    return [
        {
            "text": hit.fields["text"],
            "filename": hit.fields["filename"],
            "page_number": int(hit.fields["page_number"]),
            "score": float(hit.score),
        }
        for hit in res.result.hits
        if hit.score >= settings.min_relevance_score
    ]
