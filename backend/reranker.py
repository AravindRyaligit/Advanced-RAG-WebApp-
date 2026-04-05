from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from backend.config import settings


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load and cache the BGE cross-encoder reranker model."""
    return CrossEncoder(settings.reranker_model, max_length=512)


def rerank(query: str, documents: list[Document], top_n: int | None = None) -> list[Document]:
    """
    Re-rank retrieved documents using the cross-encoder.

    Takes the query and candidate documents, scores each (query, doc) pair,
    and returns the top_n documents sorted by relevance.
    """
    if not documents:
        return []

    top_n = top_n or settings.rerank_top_n
    model = get_reranker()

    pairs = [[query, doc.page_content] for doc in documents]
    scores = model.predict(pairs)

    scored_docs = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True,
    )

    return [doc for _, doc in scored_docs[:top_n]]
