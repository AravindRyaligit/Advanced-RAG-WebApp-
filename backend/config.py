from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""

    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    chunk_size: int = 1000
    chunk_overlap: int = 200

    retrieval_top_k: int = 10
    rerank_top_n: int = 5

    chroma_persist_dir: str = str(BASE_DIR / "data" / "chroma_db")
    chroma_collection: str = "rag_documents"

    upload_dir: str = str(BASE_DIR / "uploads")


settings = Settings()
