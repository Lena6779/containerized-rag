# These tests replace Ollama and ChromaDB calls with fakes,
# so they run in CI without a model or an embedding download
import rag
import requests
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "model" in response.json()


def test_health(monkeypatch):
    monkeypatch.setattr(rag, "check_ollama", lambda: True)
    monkeypatch.setattr(rag, "count", lambda: 3)
    data = client.get("/health").json()
    assert data["status"] == "healthy"
    assert data["ollama"] == "connected"
    assert data["documents"] == 3


def test_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "document_count" in data
    assert "chunk_count" in data


def test_ingest(monkeypatch):
    result = {"files": ["sample.txt"], "chunks": 5}
    monkeypatch.setattr(rag, "ingest_documents", lambda: result)
    response = client.post("/ingest")
    assert response.status_code == 200
    assert response.json() == result


def test_ask_returns_answer_and_sources(monkeypatch):
    source = {"source": "sample.txt", "text": "Docker runs containers.", "distance": 0.2}
    monkeypatch.setattr(rag, "count", lambda: 1)
    monkeypatch.setattr(rag, "retrieve", lambda question: [source])
    monkeypatch.setattr(rag, "generate", lambda question, sources: "It runs containers.")
    data = client.post("/ask", json={"question": "What does Docker do?"}).json()
    assert data == {"answer": "It runs containers.", "sources": [source], "confidence": "high"}


def test_ask_without_documents(monkeypatch):
    monkeypatch.setattr(rag, "count", lambda: 0)
    data = client.post("/ask", json={"question": "Anything?"}).json()
    assert data["sources"] == []
    assert "No documents" in data["answer"]


def test_ask_without_matching_documents(monkeypatch):
    monkeypatch.setattr(rag, "count", lambda: 1)
    monkeypatch.setattr(rag, "retrieve", lambda question: [])
    data = client.post("/ask", json={"question": "Off topic?"}).json()
    assert data["sources"] == []
    assert "don't have information" in data["answer"]


def test_ask_when_ollama_is_down(monkeypatch):
    def fail(question):
        raise requests.ConnectionError

    monkeypatch.setattr(rag, "answer", fail)
    response = client.post("/ask", json={"question": "Hello?"})
    assert response.status_code == 503


def test_ask_requires_a_question():
    assert client.post("/ask", json={}).status_code == 422


def test_chunk_text_splits_paragraphs():
    assert rag.chunk_text("First.\n\n  Second.  \n\n\n") == ["First.", "Second."]
