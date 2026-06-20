"""Re-chunk and re-embed existing documents after a chunking change.

Downloads each PDF from Supabase Storage, re-runs the (now updated) chunker,
drops the old Pinecone vectors, and upserts the new ones.

Usage (from backend/):
    python -m scripts.reindex <agent_id>   # one agent's documents
    python -m scripts.reindex              # all documents marked ready
"""

import os
import sys
from uuid import UUID

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings  # noqa: E402
from database import get_supabase  # noqa: E402
from services import rag  # noqa: E402


def reindex(agent_id: str, document_id: str, filename: str) -> None:
    db = get_supabase()
    storage_path = f"{agent_id}/{document_id}.pdf"
    pdf_bytes = db.storage.from_(settings.supabase_storage_bucket).download(storage_path)

    pages = rag.extract_pdf_pages(pdf_bytes)
    chunks = [
        (page_num, chunk)
        for page_num, page_text in pages
        for chunk in rag.chunk_text(page_text)
    ]

    rag.delete_document(UUID(agent_id), UUID(document_id))
    rag.upsert_chunks(UUID(agent_id), UUID(document_id), filename, chunks)
    print(f"  {filename}: {len(pages)} pages -> {len(chunks)} chunks")


def main() -> None:
    db = get_supabase()
    query = db.table("documents").select("id,agent_id,filename").eq("status", "ready")
    if len(sys.argv) > 1:
        query = query.eq("agent_id", sys.argv[1])
    rows = query.execute().data or []

    if not rows:
        print("no documents to reindex")
        return

    for row in rows:
        print(f"{row['agent_id']}")
        reindex(row["agent_id"], row["id"], row["filename"])


if __name__ == "__main__":
    main()
