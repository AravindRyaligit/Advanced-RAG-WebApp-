# RAG Agent - PDF Q&A

A Retrieval-Augmented Generation agent that lets you upload PDF documents and ask questions about their content. Built with LangChain, Groq (Llama 3.3 70B), ChromaDB, and a BGE reranker for improved accuracy.

## Architecture

| Component      | Technology                  |
| -------------- | --------------------------- |
| LLM            | Llama 3.3 70B via Groq API  |
| Embeddings     | BAAI/bge-base-en-v1.5 (local) |
| Re-ranker      | BAAI/bge-reranker-base (local) |
| Vector Store   | ChromaDB (persistent)       |
| Framework      | LangChain                   |
| Backend API    | FastAPI                     |
| Frontend       | Streamlit                   |

## Setup

### 1. Install dependencies

```bash
cd A:\Projects\RAG
pip install -r requirements.txt
```

### 2. Configure environment

Copy the example env file and add your Groq API key:

```bash
copy .env.example .env
```

Edit `.env` and set your key:

```
GROQ_API_KEY=gsk_your_actual_key
```

Get a free API key at https://console.groq.com

### 3. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

The first run will download the embedding model (~440 MB) and reranker model (~1.1 GB) from HuggingFace. Subsequent runs use the cached models.

### 4. Start the frontend

In a separate terminal:

```bash
streamlit run frontend/app.py
```

The Streamlit app opens at http://localhost:8501.

## Usage

1. Open the Streamlit app in your browser.
2. Upload a PDF using the sidebar.
3. Wait for indexing to complete.
4. Ask questions in the chat interface.
5. View source references in the expandable "View Sources" section under each answer.

## API Endpoints

| Method   | Endpoint              | Description                      |
| -------- | --------------------- | -------------------------------- |
| `POST`   | `/upload`             | Upload and index a PDF           |
| `POST`   | `/query`              | Ask a question                   |
| `GET`    | `/documents`          | List indexed documents           |
| `DELETE` | `/documents/{doc_id}` | Remove a document from the index |
| `GET`    | `/health`             | Health check                     |

## Project Structure

```
RAG/
  backend/
    __init__.py
    main.py                 # FastAPI application
    config.py               # Settings (env vars, model config)
    document_processor.py   # PDF parsing and text chunking
    embeddings.py           # BGE embedding model
    vectorstore.py          # ChromaDB operations
    reranker.py             # Cross-encoder reranker
    rag_chain.py            # RAG pipeline (retrieve -> rerank -> generate)
    schemas.py              # Pydantic request/response models
  frontend/
    app.py                  # Streamlit chat UI
  data/
    chroma_db/              # Persistent vector store
  uploads/                  # Stored PDF files
  requirements.txt
  .env.example
  README.md
```
