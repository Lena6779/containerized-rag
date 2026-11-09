# FastAPI routes for the RAG API; the pipeline itself lives in rag.py
import rag
import requests
from config import settings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class Question(BaseModel):
    """Request body for POST /ask."""
    question: str


@app.get("/")
def root():
    """Confirm the API is running and show the model name."""
    return {"message": "RAG API running", "model": settings.MODEL_NAME}


# Used by the Docker healthcheck and the frontend sidebar
@app.get("/health")
def health():
    """Report Ollama status and the number of stored chunks."""
    return {
        "status": "healthy",
        "ollama": "connected" if rag.check_ollama() else "unavailable",
        "ollama_url": settings.OLLAMA_URL,
        "documents": rag.count(),
    }


@app.get("/stats")
def stats():
    """Report document and chunk counts for the collection."""
    return rag.get_stats()


# Reload every .txt file from docs/ into ChromaDB
@app.post("/ingest")
def ingest():
    """Reload every .txt and .md file from docs/ into ChromaDB."""
    return rag.ingest_documents()


@app.post("/ask")
def ask(q: Question):
    """Answer a question and return its sources and confidence level."""
    try:
        return rag.answer(q.question)
    except requests.RequestException as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not get an answer from Ollama. Is {settings.MODEL_NAME} pulled?",
        ) from error
