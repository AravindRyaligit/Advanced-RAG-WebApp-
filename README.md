# RAG-based Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents (PDFs, TXTs) and ask questions based on their content. The system uses embeddings and a vector store to retrieve relevant information and an LLM to generate accurate, context-aware answers.

## 🚀 Features
- **Document Ingestion:** Upload and process PDF and TXT documents.
- **Smart Retrieval:** Vector-based search to find the most relevant document chunks.
- **RAG Pipeline:** Combines retrieved context with an LLM to answer questions accurately.
- **Interactive UI:** A clean, modern frontend built with React and Vite for a seamless user experience.

## 🛠️ Technology Stack
- **Backend:** FastAPI (Python), LangChain (or equivalent RAG framework), FAISS (Vector Store)
- **Frontend:** React, Vite, Lucide React (Icons)
- **Deployment:** Ready for local development with Uvicorn and Vite Dev Server.

## 📁 Project Structure
- `backend/` - Contains the FastAPI application, RAG pipeline logic, and vector store.
  - `main.py` - FastAPI application exposing `/upload`, `/chat`, and `/documents` endpoints.
  - `rag_pipeline.py` - Core logic for document ingestion, embedding, and answering questions.
- `frontend/` - Contains the React user interface.
  - `src/App.jsx` - Main React component for the chat interface.
  - `package.json` - Frontend dependencies and scripts.

## ⚙️ Setup & Installation

### Backend Setup
1. Navigate to the `backend/` directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables (e.g., API keys) in a `.env` file.
5. Run the FastAPI development server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup
1. Navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```
2. Install the required dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```

## 📝 Usage
1. Open the frontend in your browser (usually `http://localhost:5173`).
2. Upload a PDF or TXT document using the provided interface.
3. Start asking questions related to the uploaded document in the chat interface!

## 📜 License
This project is open-source and free to use.
