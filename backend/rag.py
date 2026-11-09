# RAG pipeline: load documents, retrieve relevant chunks, and generate answers
from pathlib import Path

import chromadb
import requests
from config import settings

# PersistentClient saves data to CHROMA_PATH, which is a named volume in Docker
client = chromadb.PersistentClient(path=settings.CHROMA_PATH)


def _open_collection():
    """Open the ChromaDB collection, creating it if it doesn't exist."""
    # Cosine distance: 0 = identical, 2 = opposite
    return client.get_or_create_collection(
        settings.COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}}
    )


collection = _open_collection()


def count() -> int:
    """Return the number of chunks stored in the collection."""
    return collection.count()


def check_ollama() -> bool:
    """Return True if the Ollama server responds."""
    try:
        response = requests.get(f"{settings.OLLAMA_URL}/api/tags", timeout=3)
    except requests.RequestException:
        return False
    return response.status_code == 200


def get_stats() -> dict:
    """Return document and chunk counts, source files, collection, and model."""
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    sources = sorted({(meta or {}).get("source", "unknown") for meta in metadatas})
    return {
        "document_count": len(sources),
        "chunk_count": len(metadatas),
        "sources": sources,
        "collection": settings.COLLECTION_NAME,
        "model": settings.MODEL_NAME,
    }


def is_heading(paragraph: str) -> bool:
    """Treat short text that doesn't end in a period as a heading."""
    return len(paragraph.split()) <= 8 and not paragraph.endswith(".")


def chunk_text(text: str) -> list[str]:
    """Split text into paragraphs and attach each section's heading to them."""
    chunks = []
    heading = ""
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if is_heading(paragraph):
            heading = paragraph
        else:
            chunks.append(f"{heading}\n\n{paragraph}" if heading else paragraph)
    return chunks


def ingest_documents() -> dict:
    """Replace the collection with every .txt and .md file in DOCS_DIRECTORY."""
    global collection

    docs_dir = Path(settings.DOCS_DIRECTORY)
    files = sorted(list(docs_dir.glob("*.txt")) + list(docs_dir.glob("*.md")))

    ids, texts, metadatas = [], [], []
    for file in files:
        for i, chunk in enumerate(chunk_text(file.read_text(encoding="utf-8"))):
            ids.append(f"{file.name}_{i}")
            texts.append(chunk)
            metadatas.append({"source": file.name, "chunk_index": i})

    # Existing data is only replaced once there is new text to index
    if texts:
        client.delete_collection(settings.COLLECTION_NAME)
        collection = _open_collection()
        collection.add(ids=ids, documents=texts, metadatas=metadatas)

    return {"files": [file.name for file in files], "chunks": len(texts)}


def retrieve(question: str) -> list[dict]:
    """Return the closest chunks that are within the confidence threshold."""
    results = collection.query(
        query_texts=[question],
        n_results=min(settings.MAX_RESULTS, collection.count()),
    )

    sources = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        if settings.DEBUG:
            print(f"[ask] distance={dist:.3f} doc={doc[:60]!r}")
        if dist <= settings.CONFIDENCE_THRESHOLD:
            sources.append(
                {
                    "source": (meta or {}).get("source", "unknown"),
                    "chunk_index": (meta or {}).get("chunk_index"),
                    "text": doc,
                    "distance": round(dist, 3),
                }
            )
    return sources


def generate(question: str, sources: list[dict]) -> str:
    """Ask Ollama to answer using only the retrieved chunks."""
    context = "\n\n".join(source["text"] for source in sources)
    response = requests.post(
        f"{settings.OLLAMA_URL}/api/chat",
        json={
            "model": settings.MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer using ONLY the context provided. "
                    "If the answer isn't in the context, say you don't know.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def get_confidence(sources: list[dict]) -> str:
    """Rate retrieval confidence as high, medium, or low."""
    if not sources:
        return "low"

    best_distance = min(source["distance"] for source in sources)
    if best_distance <= settings.HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if best_distance <= settings.MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def answer(question: str) -> dict:
    """Run the full RAG flow: retrieve, filter, then generate."""
    if count() == 0:
        return {
            "answer": "No documents yet. Run POST /ingest first.",
            "sources": [],
            "confidence": "none",
        }

    sources = retrieve(question)
    if not sources:
        return {
            "answer": "I don't have information about that in my documents.",
            "sources": [],
            "confidence": "none",
        }

    return {
        "answer": generate(question, sources),
        "sources": sources,
        "confidence": get_confidence(sources),
    }
