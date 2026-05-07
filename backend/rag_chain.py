from __future__ import annotations

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from backend.config import settings
from backend.vectorstore import similarity_search
from backend.reranker import rerank
from backend.guardrails import check_input, check_output


SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based on the provided context.
Use ONLY the context below to answer. If the context does not contain enough
information to answer the question, say so clearly -- do not make things up.

When referencing information, mention the source filename and page number if available.

Context:
{context}"""

USER_PROMPT = "{question}"


def _get_llm() -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
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
    is_safe, reason = check_input(question)
    if not is_safe:
        return {
            "answer": f"I cannot answer this question. The request violates safety policies. ({reason})",
            "sources": []
        }

    retrieved_docs = similarity_search(question)

    reranked_docs = rerank(question, retrieved_docs)

    context = _format_context(reranked_docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    chain = prompt | _get_llm() | StrOutputParser()

    answer = chain.invoke({"context": context, "question": question})

    is_safe_output, output_reason = check_output(question, answer)
    if not is_safe_output:
        return {
            "answer": f"The generated response was blocked due to safety policy violations. ({output_reason})",
            "sources": _build_sources(reranked_docs)
        }

    return {
        "answer": answer,
        "sources": _build_sources(reranked_docs),
    }
