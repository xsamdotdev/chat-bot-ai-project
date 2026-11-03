# RAG Document Chatbot — Backend

A FastAPI + LangChain backend that lets users upload large documents, then ask
questions against them. Every query is answered with **retrieval-augmented
generation (RAG)**: the question is embedded, the most relevant chunks are
vector-searched from a local FAISS index, and the retrieved context is sent to
an LLM which produces a grounded answer with source citations.

Frontend is intentionally not included yet — this repo currently ships the API only.

## Features

- 📄 Upload **PDF, DOCX, TXT, Markdown** (up to 50 MB by default, configurable)
- ✂️ Automatic parsing → chunking (`RecursiveCharacterTextSplitter`)
- 🔍 Vector similarity search over a **persistent local FAISS** index
- 🤖 Chat answers via LangChain, grounded in retrieved context, with **source citations**
- 🔌 **OpenAI** (or any OpenAI-compatible endpoint), **Mistral AI**, or **fully-local Ollama** — switched with one `.env` variable
- 🧪 `fake` provider for fully offline development/smoke tests (no keys needed)
- 🗂️ Per-embedding-model index isolation + document manifest (list / delete documents)
- ✅ Pytest suite that runs offline, OpenAPI docs at `/docs`

## Architecture

```
┌──────────┐   upload   ┌─────────────┐   chunks   ┌──────────────┐
│  client  │ ─────────► │  FastAPI    │ ─────────► │  FAISS index │
│ (later)  │            │  /api       │            │  on disk     │
└──────────┘            └─────┬───────┘            └──────────────┘
     ▲                        │ query                     ▲
     │ answer + sources       ▼                           │ top-k
     └──────────────── ┌─────────────┐  context  ┌────────┴───────┐
                       │  LangChain  │ ◄──────── │  vector search │
                       │  (prompt +  │           └────────────────┘
                       │   LLM)      │
                       └─────────────┘
```

Ingestion: `file → loader → chunker → embeddings → FAISS (+ manifest.json)`
Query: `question → embed → similarity_search(k) → prompt(context, history?, question) → LLM → answer + sources`

## Run with Docker (full stack: Ollama + backend)

Prerequisites: Docker Desktop / Docker Engine with Compose v2, ~6 GB free disk.

```bash
docker compose up --build
```

This starts three things:

| Service | Purpose |
|---|---|
| `ollama` | Local LLM runtime; models persist in the `ollama-data` volume |
| `ollama-models` | One-shot init job — pulls `llama3.1` + `nomic-embed-text` (several GB on first run; idempotent afterwards) |
| `backend` | FastAPI + LangChain API on http://localhost:8000 (docs at `/docs`) |

**Data persistence:** uploads + FAISS index + manifest live in the `chatbot-data`
volume (`/data` inside the container). `docker compose down` keeps them;
`docker compose down -v` deletes them.

**Useful variations** (compose reads a repo-root `.env` automatically):

```bash
# OpenAI instead of Ollama — starts only the backend, no model downloads
PROVIDER=openai OPENAI_API_KEY=sk-... docker compose up --build backend --no-deps

# Mistral AI instead of Ollama — same, starts only the backend
PROVIDER=mistral MISTRAL_API_KEY=... docker compose up --build backend --no-deps

# Fully offline fake mode (smoke tests, frontend dev)
PROVIDER=fake docker compose up --build backend --no-deps

# Different Ollama models (first pull happens on up)
OLLAMA_CHAT_MODEL=qwen2.5 OLLAMA_EMBEDDING_MODEL=nomic-embed-text docker compose up --build

# GPU passthrough for Ollama — uncomment the deploy: block in compose.yaml first
```

> **No separate vector DB service is required** — FAISS is embedded in the backend
> and persists in `chatbot-data`. A managed vector DB (Qdrant/pgvector) would only
> be needed for multi-instance deployments; see Operational notes.

Backend env (`PROVIDER`, `OPENAI_*`, `OLLAMA_*`, `CHUNK_SIZE`, …) can all be
set on the `backend` service in `compose.yaml` or via a root `.env`.

## Project layout

```
backend/
  app/
    main.py                 # app factory, CORS, exception handler, /docs
    config.py               # pydantic-settings (.env) — single source of truth
    schemas.py              # request/response models
    api/routes/             # health, documents, chat
    services/
      document_loader.py    # PDF/DOCX/TXT/MD → LangChain Documents
      chunking.py           # RecursiveCharacterTextSplitter wrapper
      embeddings.py         # provider factory (openai | ollama | mistral | fake)
      llm.py                # provider factory + offline EchoChatModel
      vectorstore.py        # FAISS manager: add/search/delete + manifest
      rag.py                # retrieve → prompt → LLM → answer + sources
    storage/                # runtime data (gitignored): uploads + indexes
  tests/                    # offline pytest suite
  requirements.txt
  .env.example
```

## Prerequisites

- Python 3.11+ (developed/tested on 3.14)
- One of:
  - an **OpenAI API key** (`PROVIDER=openai`), or
  - a **Mistral AI API key** (`PROVIDER=mistral`), or
  - **Ollama** running locally with pulled models (`PROVIDER=ollama`), or
  - nothing at all (`PROVIDER=fake` — offline dev stub)

## Quickstart

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env     # then edit .env (see below)
uvicorn app.main:app --reload
```

API is now at <http://localhost:8000> — interactive docs at <http://localhost:8000/docs>.

### Option A — OpenAI

```env
PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Works with any OpenAI-compatible endpoint too:

```env
OPENAI_BASE_URL=https://api.groq.com/openai/v1
```

### Option B — Ollama (fully local, free)

```powershell
ollama pull llama3.1
ollama pull nomic-embed-text
```

```env
PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### Option C — Mistral AI

```env
PROVIDER=mistral
MISTRAL_API_KEY=...
MISTRAL_CHAT_MODEL=mistral-small
MISTRAL_EMBEDDING_MODEL=mistral-embed
```

Chat and embeddings both go through the Mistral API (embeddings: `mistral-embed`,
1024-dim). To point at a self-hosted/proxy, set `MISTRAL_BASE_URL` (defaults to
`https://api.mistral.ai/v1/`). Get a key at <https://console.mistral.ai/>.

### Option D — fake (offline smoke tests / frontend dev without AI)

```env
PROVIDER=fake
```

Zero setup: uploads are embedded with deterministic fake vectors and answers
come from an echo stub. Use it to verify plumbing, not answer quality.

## API reference

Base URL: `http://localhost:8000` — all endpoints live under `/api`.

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness + active provider/models + docs indexed |
| POST | `/api/documents/upload` | Multipart upload & ingest (201) |
| GET | `/api/documents` | List ingested documents for the current embedding model |
| DELETE | `/api/documents/{doc_id}` | Remove a document and its vectors |
| POST | `/api/chat` | RAG query → answer + sources |

Full schemas are in Swagger UI (`/docs`). Examples with curl:

```bash
# 1. Upload a document
curl -F "file=@./big-report.pdf" http://localhost:8000/api/documents/upload

# 2. List what's in the index
curl http://localhost:8000/api/documents

# 3. Ask a question
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings?", "top_k": 4}'

# 4. With conversation history (optional)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can you elaborate on that?",
    "history": [
      {"role": "user", "content": "Summarize the report"},
      {"role": "assistant", "content": "The report covers ..."}
    ]
  }'

# 5. Delete a document
curl -X DELETE http://localhost:8000/api/documents/<doc_id>
```

### Upload response (201)

```json
{
  "message": "Document ingested successfully.",
  "document": {
    "doc_id": "9f2c...",
    "filename": "big-report.pdf",
    "chunks": 148,
    "size_bytes": 1834221,
    "uploaded_at": "2026-11-03T09:00:00+00:00"
  }
}
```

### Chat response (200)

```json
{
  "answer": "The main finding is ... [Source 1]",
  "sources": [
    {
      "doc_id": "9f2c...",
      "filename": "big-report.pdf",
      "chunk_index": 17,
      "page": 4,
      "snippet": "...",
      "score": 0.41
    }
  ],
  "provider": "openai",
  "model": "gpt-4o-mini"
}
```

**Error codes:** `400` empty file · `404` unknown doc_id · `409` chat before any upload (or after all docs were deleted) ·
`413` file too large · `415` unsupported extension · `422` validation · `502` LLM/retrieval failure · `503` provider misconfigured.

## Configuration

Everything is set via `backend/.env` (see `.env.example`):

| Variable | Default | Notes |
|---|---|---|
| `PROVIDER` | `openai` | `openai` \| `ollama` \| `mistral` \| `fake` |
| `OPENAI_API_KEY` | — | required for `openai` |
| `OPENAI_BASE_URL` | — | for OpenAI-compatible endpoints |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | any chat model on the endpoint |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | changing it switches the index dir |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `OLLAMA_CHAT_MODEL` | `llama3.1` | |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | |
| `MISTRAL_API_KEY` | — | required for `mistral` |
| `MISTRAL_BASE_URL` | — | defaults to `https://api.mistral.ai/v1/`; set for proxies/self-hosted |
| `MISTRAL_CHAT_MODEL` | `mistral-small` | any chat model on the Mistral API |
| `MISTRAL_EMBEDDING_MODEL` | `mistral-embed` | changing it switches the index dir |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | overlap must be < chunk size |
| `TOP_K` | `4` | chunks retrieved per query (per-request override allowed) |
| `MAX_UPLOAD_SIZE_MB` | `50` | streamed to disk, enforced mid-upload |
| `CORS_ORIGINS` | `*` | comma-separated; set to your frontend origin in prod |
| `UPLOAD_DIR` / `VECTORSTORE_DIR` | `backend/storage/...` | runtime data, gitignored |

> **Switching providers:** the FAISS index lives in `storage/vectorstore/<embedding-model>/`,
> so OpenAI, Ollama and Mistral vectors never mix. Documents ingested under one model don't
> appear under another — re-upload after switching (or keep `VECTORSTORE_DIR` per env).

## Running the tests

```powershell
cd backend
.venv\Scripts\activate
pip install -r requirements-dev.txt
pytest
```

All tests run offline using the `fake` provider and temporary storage — no API
keys, no Ollama, no network.

## Operational notes & limits

- **Vector store source.** FAISS ships via `langchain-community` (in maintenance mode upstream); tests pin it and coverage is green. The `VectorStoreManager` is the swap point for a supported backend (`langchain-faiss` is currently an empty placeholder, and `langchain-community` loaders/FAISS will need migration when replacements land).
- **Deletion** removes vectors by id (`<doc_id>:<chunk_index>`) via FAISS `remove_ids` and updates the manifest; the stored upload file is also removed.
- **Single-process storage.** FAISS + JSON manifest are fine for one uvicorn worker. For multi-worker/multi-instance deployments, move to a server vector DB (Qdrant/pgvector) — the `VectorStoreManager` interface is the seam.
- **Scanned PDFs** (images, no text layer) need OCR before ingest; text-based PDFs work out of the box.
- **History** is client-supplied (`history` array) — the server is stateless; your frontend owns conversation state.
- Changing `CHUNK_SIZE`/`OVERLAP` only affects future uploads; re-upload to re-chunk.

## Roadmap

- [ ] Frontend (React) — next
- [ ] Streaming chat responses (SSE)
- [ ] Server vector DB option (Qdrant / pgvector)
- [ ] OCR fallback for scanned PDFs
- [ ] Auth (API key / JWT) — required before exposing beyond localhost
