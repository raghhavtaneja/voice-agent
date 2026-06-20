"""One-time setup: create the Pinecone integrated index.

Usage (from the ``backend/`` directory, with ``.env`` populated):

    python -m scripts.setup_pinecone
"""

from pinecone import Pinecone

from config import settings


def main() -> None:
    pc = Pinecone(api_key=settings.pinecone_api_key)

    if pc.has_index(settings.pinecone_index):
        print(f"index {settings.pinecone_index!r} already exists; nothing to do")
        return

    pc.create_index_for_model(
        name=settings.pinecone_index,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        embed={
            "model": settings.pinecone_embed_model,
            "field_map": {"text": "text"},  # the record field that gets embedded
        },
    )
    print(
        f"created index {settings.pinecone_index!r} "
        f"({settings.pinecone_embed_model}, {settings.pinecone_cloud}/{settings.pinecone_region})"
    )


if __name__ == "__main__":
    main()
