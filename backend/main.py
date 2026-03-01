import os
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline import ingest_document, ask_question
from pydantic import BaseModel

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

TRACKER_FILE = "uploaded_docs.json"

def get_tracked_docs():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def add_tracked_doc(filename: str):
    docs = get_tracked_docs()
    if filename not in docs:
        docs.append(filename)
        with open(TRACKER_FILE, "w") as f:
            json.dump(docs, f)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    # Save the file temporarily
    file_location = f"temp_{file.filename}"
    try:
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Ingest document into vector store
        ingest_document(file_location, file.filename)
        
        # Track the successfully uploaded document
        add_tracked_doc(file.filename)
        
        return {"message": f"Successfully ingested {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

@app.get("/documents")
async def get_documents():
    docs = get_tracked_docs()
    return {"documents": docs}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        answer = ask_question(request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok"}
