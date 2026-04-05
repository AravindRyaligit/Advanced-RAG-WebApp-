from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.schemas import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    DocumentsListResponse,
    DocumentInfo,
    DeleteResponse,
    HealthResponse,
)
from backend.document_processor import process_pdf, delete_upload
from backend.vectorstore import add_documents, list_documents, delete_by_doc_id
from backend.rag_chain import query_rag

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading embedding model (first run downloads ~440 MB)...")
    from backend.embeddings import get_embedding_model
    get_embedding_model()

    logger.info("Loading reranker model (first run downloads ~1.1 GB)...")
    from backend.reranker import get_reranker
    get_reranker()

    logger.info("Models loaded. Server is ready.")
    yield


app = FastAPI(
    title="RAG Agent API",
    description="PDF-based Retrieval-Augmented Generation with Groq Llama 70B",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF document, extract text, chunk, embed, and store in ChromaDB."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        doc_id, chunks = process_pdf(file.filename, contents)
        add_documents(chunks)
    except Exception as e:
        logger.exception("Failed to process PDF")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {e}")

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        chunk_count=len(chunks),
        message=f"Successfully indexed {len(chunks)} chunks from '{file.filename}'.",
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question against the indexed documents."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured.")

    try:
        result = query_rag(request.question)
    except Exception as e:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return QueryResponse(**result)


@app.get("/documents", response_model=DocumentsListResponse)
async def get_documents():
    """List all indexed documents."""
    docs = list_documents()
    return DocumentsListResponse(
        documents=[DocumentInfo(**d) for d in docs]
    )


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """Remove a document and its chunks from the index."""
    try:
        delete_by_doc_id(doc_id)
        delete_upload(doc_id)
    except Exception as e:
        logger.exception("Delete failed")
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

    return DeleteResponse(doc_id=doc_id, message="Document deleted successfully.")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
    )
