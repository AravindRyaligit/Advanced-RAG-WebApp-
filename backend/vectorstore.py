from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.config import settings
from backend.embeddings import get_embedding_model


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Return the persistent ChromaDB-backed vectorstore."""
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embedding_model(),
        persist_directory=str(persist_dir),
    )


def add_documents(chunks: list[Document]) -> list[str]:
    """Embed chunks and add them to the vectorstore. Returns Chroma IDs."""
    vs = get_vectorstore()
    ids = vs.add_documents(chunks)
    return ids


def similarity_search(query: str, k: int | None = None) -> list[Document]:
    """Retrieve the top-k most similar chunks for a query."""
    vs = get_vectorstore()
    return vs.similarity_search(query, k=k or settings.retrieval_top_k)


def delete_by_doc_id(doc_id: str) -> None:
    """Remove all chunks belonging to a specific document from ChromaDB."""
    vs = get_vectorstore()
    collection = vs._collection
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])


def list_documents() -> list[dict]:
    """Return a deduplicated list of indexed documents with metadata."""
    vs = get_vectorstore()
    collection = vs._collection
    all_data = collection.get(include=["metadatas"])

    docs_map: dict[str, dict] = {}
    for meta in all_data.get("metadatas", []):
        if not meta:
            continue
        doc_id = meta.get("doc_id", "unknown")
        if doc_id not in docs_map:
            docs_map[doc_id] = {
                "doc_id": doc_id,
                "source": meta.get("source", "unknown"),
                "chunk_count": 0,
            }
        docs_map[doc_id]["chunk_count"] += 1

    return list(docs_map.values())
