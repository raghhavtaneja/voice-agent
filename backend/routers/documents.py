"""Document upload + delete — HTTP layer only."""

from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Response, UploadFile

from config import settings
from database import get_supabase
from models.document import Document, DocumentStatus
from services import rag

router = APIRouter(prefix="/agents/{agent_id}/documents", tags=["documents"])


@router.post("", response_model=Document, status_code=201)
async def upload_document(agent_id: UUID, file: UploadFile) -> Document:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "file must be a PDF")

    db = get_supabase()

    agent = db.table("agents").select("id").eq("id", str(agent_id)).maybe_single().execute()
    if not agent.data:
        raise HTTPException(404, "agent not found")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "empty file")

    document_id = uuid4()
    storage_path = f"{agent_id}/{document_id}.pdf"

    db.storage.from_(settings.supabase_storage_bucket).upload(
        path=storage_path,
        file=pdf_bytes,
        file_options={"content-type": "application/pdf"},
    )
    file_url = db.storage.from_(settings.supabase_storage_bucket).get_public_url(storage_path)

    db.table("documents").insert(
        {
            "id": str(document_id),
            "agent_id": str(agent_id),
            "filename": file.filename,
            "file_url": file_url,
            "status": DocumentStatus.processing.value,
        }
    ).execute()

    # TODO: move parsing/embedding to a background worker once we have one.
    try:
        pages = rag.extract_pdf_pages(pdf_bytes)
        chunks: list[tuple[int, str]] = [
            (page_num, chunk)
            for page_num, page_text in pages
            for chunk in rag.chunk_text(page_text)
        ]
        rag.upsert_chunks(agent_id, document_id, file.filename, chunks)
    except Exception as exc:
        db.table("documents").update({"status": DocumentStatus.failed.value}).eq(
            "id", str(document_id)
        ).execute()
        raise HTTPException(500, f"failed to process PDF: {exc}") from exc

    res = (
        db.table("documents")
        .update({"status": DocumentStatus.ready.value})
        .eq("id", str(document_id))
        .execute()
    )
    return Document(**res.data[0])


@router.delete("/{document_id}", status_code=204, response_class=Response)
def delete_document(agent_id: UUID, document_id: UUID) -> Response:
    """Cascade-delete a document: Pinecone chunks, storage file, then DB row."""
    db = get_supabase()

    exists = (
        db.table("documents")
        .select("id")
        .eq("id", str(document_id))
        .eq("agent_id", str(agent_id))
        .limit(1)
        .execute()
    )
    if not exists.data:
        raise HTTPException(404, "document not found")

    # Pinecone first — idempotent, fine if there were never any chunks.
    rag.delete_document(agent_id, document_id)

    # Storage path matches what we upload to.
    storage_path = f"{agent_id}/{document_id}.pdf"
    db.storage.from_(settings.supabase_storage_bucket).remove([storage_path])

    db.table("documents").delete().eq("id", str(document_id)).execute()

    return Response(status_code=204)
