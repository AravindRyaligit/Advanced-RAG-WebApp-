from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from backend.config import settings
from backend.vectorstore import similarity_search
from backend.reranker import rerank


SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based on the provided context.
Use ONLY the context below to answer. If the context does not contain enough
information to answer the question, say so clearly -- do not make things up.

When referencing information, mention the source filename and page number if available.

Context:
{context}"""

USER_PROMPT = "{question}"


def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model_name=settings.llm_model,
        temperature=settings.llm_temperature,
    )


def _format_context(docs: list[Document]) -> str:
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] (Source: {source}, Page: {page})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_sources(docs: list[Document]) -> list[dict]:
    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", 0),
            "content": doc.page_content[:300],
        }
        for doc in docs
    ]


def query_rag(question: str) -> dict:
    """
    Full RAG pipeline: retrieve -> rerank -> generate.

    Returns dict with 'answer' and 'sources'.
    """
    retrieved_docs = similarity_search(question)

    reranked_docs = rerank(question, retrieved_docs)

    context = _format_context(reranked_docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    chain = prompt | _get_llm() | StrOutputParser()

    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "sources": _build_sources(reranked_docs),
    }
