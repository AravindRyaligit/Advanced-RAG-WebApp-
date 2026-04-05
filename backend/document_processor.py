from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from backend.config import settings


def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def save_upload(filename: str, file_bytes: bytes) -> tuple[str, Path]:
    """Persist an uploaded PDF and return (doc_id, saved_path)."""
    doc_id = _file_hash(file_bytes)
    dest = Path(settings.upload_dir) / f"{doc_id}_{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(file_bytes)
    return doc_id, dest


def extract_text_from_pdf(pdf_path: Path) -> list[Document]:
    """Read a PDF and return one Document per page with metadata."""
    reader = PdfReader(str(pdf_path))
    docs: list[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num,
                    },
                )
            )
    return docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split page-level documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def process_pdf(filename: str, file_bytes: bytes) -> tuple[str, list[Document]]:
    """End-to-end: save PDF -> extract text -> chunk. Returns (doc_id, chunks)."""
    doc_id, saved_path = save_upload(filename, file_bytes)
    pages = extract_text_from_pdf(saved_path)
    chunks = chunk_documents(pages)
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
    return doc_id, chunks


def delete_upload(doc_id: str) -> None:
    """Remove uploaded PDF file(s) matching the given doc_id."""
    upload_dir = Path(settings.upload_dir)
    if upload_dir.exists():
        for f in upload_dir.iterdir():
            if f.name.startswith(doc_id):
                f.unlink(missing_ok=True)
