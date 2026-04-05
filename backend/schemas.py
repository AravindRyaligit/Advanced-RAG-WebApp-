from __future__ import annotations

from pydantic import BaseModel


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    message: str


class QueryRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    source: str
    page: int
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    chunk_count: int


class DocumentsListResponse(BaseModel):
    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    doc_id: str
    message: str


class HealthResponse(BaseModel):
    status: str
    llm_model: str
    embedding_model: str
    reranker_model: str
