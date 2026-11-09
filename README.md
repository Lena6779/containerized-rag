# containerized-rag

A containerized RAG application. A Streamlit chat interface talks to a FastAPI backend, which retrieves relevant text from ChromaDB and generates answers with a local Ollama model.

## Architecture

```
Browser -> frontend (Streamlit :8501) -> backend (FastAPI :8000) -> ollama (:11434)
                                              |
                                          ChromaDB (chroma_data volume)
```

| Service | What it does | Port |
|---|---|---|
| `frontend` | Streamlit chat interface | `8501` |
| `backend` | FastAPI RAG API with embedded ChromaDB | `8000` |
| `ollama` | Local LLM server | `11434` |

Named volumes keep ChromaDB data (`chroma_data`) and downloaded models (`ollama_data`) after containers are removed. `backend/docs` is mounted into the backend, so new files can be ingested without rebuilding the image.

When a user asks a question:

1. The frontend sends it to the backend's `POST /ask` endpoint.
2. The backend finds the closest chunks in ChromaDB and drops any beyond `CONFIDENCE_THRESHOLD`.
3. The backend builds a prompt from the system instructions, retrieved chunks, and question, then sends it to Ollama.
4. The backend returns the answer, its sources, and a confidence level, and the frontend displays them.

## Project structure

```
containerized-rag/
├── docker-compose.yml
├── .env                 # local settings (not committed)
├── .env.example         # settings template
├── .github/workflows/ci.yml
├── backend/
│   ├── Dockerfile
│   ├── main.py          # API routes
│   ├── rag.py           # ingest, retrieve, and generate
│   ├── config.py        # environment-based settings
│   ├── requirements.txt
│   ├── docs/            # document corpus
│   └── tests/test_api.py
└── frontend/
    ├── Dockerfile
    ├── app.py           # Streamlit chat interface
    └── requirements.txt
```

## Getting started

You need Docker Desktop running. If Ollama is installed on your computer, quit it first so port `11434` is free.

1. Create your settings file (on Windows Command Prompt, use `copy .env.example .env`):

   ```bash
   cp .env.example .env
   ```

2. Build and start the stack:

   ```bash
   docker compose up --build
   ```

3. In a second terminal, download the model. This is only needed once:

   ```bash
   docker compose exec ollama ollama pull llama3.2:1b
   ```

4. Open http://localhost:8501, click **Re-index Documents**, and ask a question.

The first re-index takes longer because ChromaDB downloads its embedding model.

## Usage

1. Click **Re-index Documents** in the sidebar the first time you run the app, and again whenever you add or change a document. The sidebar shows backend and Ollama status, plus document and chunk counts.
2. Type a question in the chat box. Questions and answers stay on the page until you refresh it or click **Clear Chat**.
3. Read the confidence level under each answer.
4. Open **Sources** to see the file, chunk number, distance, and text behind each answer. A lower distance means a closer match.

| Confidence | Closest source |
|---|---|
| `high` | Distance at or below `HIGH_CONFIDENCE_THRESHOLD` (default `0.3`) |
| `medium` | Distance at or below `MEDIUM_CONFIDENCE_THRESHOLD` (default `0.6`) |
| `low` | Any other match within `CONFIDENCE_THRESHOLD` |
| `none` | No documents are indexed, or no chunk is close enough |

Confidence rates how well the retrieved text matches the question. It does not check the wording of the model's answer.

To add a document, put a `.txt` or `.md` file in `backend/docs` and click **Re-index Documents**. You don't need to rebuild the image.

### Guardrails

- If no documents are indexed, the app asks you to ingest them.
- If no chunk is within `CONFIDENCE_THRESHOLD`, the app says it doesn't have that information and doesn't call the model.
- The system prompt tells the model to answer only from the retrieved context and to say when it doesn't know.

## Corpus and chunking

The corpus in `backend/docs` is 15 text files of course notes, about 5,300 words or 10 pages. The files cover separate topics: Python, SQL, REST APIs, FastAPI, API security, HTML and JavaScript, debugging and testing, Docker, Streamlit, embeddings, ChromaDB, chunking, prompt engineering, local LLMs, and RAG.

Documents use paragraph chunking:

1. Text is split on blank lines, so each chunk is a complete paragraph.
2. Short text of up to 8 words that doesn't end in a period is treated as a heading.
3. Each heading is attached to the paragraphs in its section, so a chunk keeps its topic and a heading is never stored alone.

This produces 108 chunks. Each chunk is stored in ChromaDB with `source` (file name) and `chunk_index` (position in the file, starting at 0) as metadata.

## Running locally without Docker

Ollama must be running on your computer with `llama3.2:1b` pulled. These commands are for PowerShell.

The first time, create the virtual environment and install the packages from the project folder:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
pip install -r frontend\requirements.txt
```

To start the backend in a new terminal:

```powershell
cd C:\path\to\containerized-rag
venv\Scripts\activate
cd backend
uvicorn main:app --reload
```

To start the frontend in a second terminal:

```powershell
cd C:\path\to\containerized-rag
venv\Scripts\activate
cd frontend
streamlit run app.py
```

`uvicorn` must run from inside `backend`, or it fails with `Could not import module "main"`. Without Docker, the app uses the localhost defaults in `backend/config.py` and `frontend/app.py`.

If you use VS Code, set `python.terminal.useEnvFile` to `false`. Otherwise new terminals load the Docker settings from `.env`, and the backend can't find Ollama or `docs/`.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Shows that the API is running and which model it uses |
| `GET` | `/health` | Shows Ollama connection status and chunk count |
| `GET` | `/stats` | Shows document count, chunk count, source files, collection, and model |
| `POST` | `/ingest` | Replaces the index with every `.txt` and `.md` file in `docs/` |
| `POST` | `/ask` | Answers a question and returns its sources and confidence level |

Interactive docs are at http://localhost:8000/docs.

`POST /ask` body:

```json
{"question": "What is a REST API?"}
```

Response (answer wording varies):

```json
{
  "answer": "A REST API is a web service that exposes resources at specific URLs called endpoints.",
  "sources": [
    {
      "source": "rest_apis.txt",
      "chunk_index": 0,
      "text": "What is a REST API?\n\nREST stands for Representational State Transfer...",
      "distance": 0.13
    }
  ],
  "confidence": "high"
}
```

If no chunk is within `CONFIDENCE_THRESHOLD`, the API says it doesn't have that information instead of guessing.

## Configuration

Docker Compose loads settings from `.env`. The backend reads them in `backend/config.py`, and the frontend receives `API_URL`.

| Variable | Purpose |
|---|---|
| `OLLAMA_URL` | Where the backend reaches Ollama |
| `API_URL` | Where the frontend reaches the backend |
| `MODEL_NAME` | Ollama model used for answers |
| `CHROMA_PATH` | ChromaDB storage path |
| `COLLECTION_NAME` | ChromaDB collection name |
| `MAX_RESULTS` | Chunks retrieved per question |
| `CONFIDENCE_THRESHOLD` | Maximum cosine distance for a match |
| `HIGH_CONFIDENCE_THRESHOLD` | Maximum closest distance for `high` confidence |
| `MEDIUM_CONFIDENCE_THRESHOLD` | Maximum closest distance for `medium` confidence |
| `DEBUG` | Prints configuration and match distances to the logs |
| `DOCS_DIRECTORY` | Folder ingested by `/ingest` |
| `OPENAI_API_KEY` | Reserved for an OpenAI fallback |

After editing `.env`, run `docker compose up` again. `docker compose restart` does not reload it.

## Tests

From `backend/`, with `requirements.txt`, `pytest`, and `httpx` installed:

```bash
python -m pytest tests/ -v
```

The tests use fakes for Ollama and retrieval, so neither needs to be running.

### Manual testing note

Answers are limited to the ingested documents. For example, "What is an api?" retrieved chunks about REST APIs, but the model answered "I don't know" because the notes define REST APIs, not APIs in general. That follows the rule to answer only from the provided context. A small model like `llama3.2:1b` can also vary between runs, so test with questions the documents answer directly, such as "What is a REST API?"

## Continuous integration

On every push and pull request to `main`, GitHub Actions runs the tests, builds both Docker images, and lints the code with Ruff.

## Useful commands

| Command | What it does |
|---|---|
| `docker compose up --build` | Rebuild and start |
| `docker compose down` | Stop and remove containers, keeping volumes |
| `docker compose ps` | Show services and health status |
| `docker compose logs backend` | Show backend logs |

`docker compose down -v` also deletes the volumes and all saved data.
