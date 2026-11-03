# PLAN.md — RAG Chatbot Backend

Working plan for the document-Q&A chatbot backend. Kept up to date as tasks land.

## Goal

Build a production-shaped FastAPI backend that:

1. Accepts large documents (PDF / DOCX / TXT / MD) via upload.
2. Parses → chunks → embeds → stores them in a persistent local vector store (FAISS).
3. On each query: embeds the query → similarity (vector) search → stuffs retrieved
   chunks into a LangChain prompt → calls the chat model → returns the answer
   **with source citations**.
4. Supports both **OpenAI** and **Ollama** (local) providers, switched via `.env`.
   Plus a `fake` provider for fully-offline dev/smoke tests (no keys, no services).

Frontend is out of scope (user will build it later). Backend lives in `backend/`.

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Framework | FastAPI + uvicorn | Requirement; async, OpenAPI docs out of the box |
| RAG orchestration | LangChain | Requirement |
| Vector store | FAISS (`faiss-cpu`), embedded, persisted to disk | Local, zero infra, single-service deploy; no separate vector DB needed |
| Containerization | Backend image + `compose.yaml` (ollama + one-shot model-pull + backend) | One command local stack; model pull idempotent |
| Backend image | python:3.14-slim, non-root user, /data volume for uploads+index | Wheels only, no compilers; secrets never in image (.dockerignore) |

| Decision | Choice | Why |
|---|---|---|
| Framework | FastAPI + uvicorn | Requirement; async, OpenAPI docs out of the box |
| RAG orchestration | LangChain (`langchain`, `langchain-community`, `langchain-openai`, `langchain-ollama`) | Requirement |
| Vector store | FAISS (`faiss-cpu`), persisted to disk | Local, zero infra, per-embedding-model index dir |
| Chunking | `RecursiveCharacterTextSplitter` (1000/200, configurable) | Solid default for prose/docs |
| Providers | `PROVIDER=openai\|ollama\|mistral\|fake` in `.env` | User chose provider switch; `fake` added for offline dev |
| Mistral AI | `langchain-mistralai`: `mistral-small` chat, `mistral-embed` embeddings, `MISTRAL_BASE_URL` override supported | Official partner package; embeddings via Mistral API as requested |
| Embeddings (openai) | `text-embedding-3-small` | Cheap, strong |
| Embeddings (ollama) | `nomic-embed-text` | Standard local embedding model |
| Chat models | `gpt-4o-mini` / `llama3.1` (configurable) | Sensible defaults, overridable |
| Index isolation | `storage/vectorstore/<embedding-model-slug>/` | Prevents mixing vectors from different embedding models |
| Doc tracking | `manifest.json` inside each index dir | List/delete documents without scanning FAISS internals |
| Delete semantics | FAISS `delete(ids)` with `{doc_id}:{chunk_index}` ids | Real removal, not just hiding from manifest |
| Chat history | Stateless; client sends optional `history` array | No server-side session state, frontend owns state |
| API prefix | `/api` | Clean frontend integration |

## Project layout

```
backend/
  app/
    __init__.py
    main.py            # app factory, lifespan, routers, exception handlers
    config.py          # pydantic-settings (.env)
    schemas.py         # Pydantic request/response models
    api/
      __init__.py
      routes/
        __init__.py
        health.py      # GET /api/health
        documents.py   # POST /api/documents/upload, GET /api/documents, DELETE /api/documents/{id}
        chat.py        # POST /api/chat  (RAG: retrieve -> generate)
    services/
      __init__.py
      document_loader.py   # extension dispatch -> LangChain loaders
      chunking.py          # RecursiveCharacterTextSplitter wrapper
      embeddings.py        # provider factory + model slug
      llm.py               # provider factory (+ offline EchoChatModel for `fake`)
      vectorstore.py       # FAISS manager: add/search/delete + manifest
      rag.py               # retrieve -> prompt -> LLM -> answer+sources
    storage/             # gitignored runtime data (uploads + index), created on boot
  tests/                 # pytest, no network/keys required (fake embeddings)
  requirements.txt       # pinned, verified in a clean venv
  .env.example
  pytest.ini (via pyproject or pytest.ini)
README.md                # repo root, user-facing docs
PLAN.md                  # this file
```

## API surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness + active provider/model |
| POST | `/api/documents/upload` | Multipart upload (pdf/docx/txt/md) → ingest |
| GET | `/api/documents` | List ingested documents (current embedding model) |
| DELETE | `/api/documents/{doc_id}` | Remove document chunks from index |
| POST | `/api/chat` | RAG query → `{answer, sources[]}` |

## Work checklist

- [x] Write PLAN.md
- [x] Scaffold backend package + config + schemas
- [x] Ingestion: loader → chunker → FAISS manager (+manifest, delete)
- [x] RAG: provider factories, retrieval chain, /api/chat
- [x] API routes + error handling + CORS + upload validation (ext/size)
- [x] requirements.txt (pinned, verified), .env.example, .gitignore
- [x] Install in clean venv; run pytest (offline) — 14 passed
- [x] Smoke test with `PROVIDER=fake` (upload → chat → delete)
- [x] Regression fix: chat after deleting all docs returns 409
- [x] README.md (setup, providers, API reference, curl examples)
- [x] Dockerfile (slim, non-root, healthcheck, /data volume) + backend/.dockerignore
- [x] compose.yaml: ollama + one-shot model pull (llama3.1, nomic-embed-text) + backend; volumes `ollama-data`, `chatbot-data`
- [x] README Docker section; compose config validated; backend image building
- [x] Mistral AI support via langchain-mistralai (chat + mistral-embed embeddings), env passthrough in compose, offline tests (17/17 green)
- [ ] Build/verify backend container end-to-end
- [ ] Full local stack verification: ollama healthy (done), nomic-embed-text pulled (done), llama3.1 pulling (in progress);
      backend will start automatically after the one-shot pull job completes

## Docker layout

- No separate vector DB service: FAISS is embedded in backend (single-instance deploys).
  Multi-instance → swap to Qdrant/pgvector behind the VectorStoreManager seam.
- `ollama` service + `ollama-models` one-shot init (pulls models once, idempotent).
- `backend` depends on ollama healthy + models pulled; `--no-deps` for OpenAI/fake mode.
- Repo-root `.env` configures compose (PROVIDER, OLLAMA_*, OPENAI_*, ports).
- Volumes: `ollama-data` (model weights), `chatbot-data` (uploads + FAISS index + manifest).

## Verification summary

- `pytest` — 14/14 green, fully offline.
- Live uvicorn smoke test (PROVIDER=fake): health → upload 47 KB .txt (75 chunks) →
  chat (3 sources, filename + chunk_index) → list → delete (75 chunks removed) →
  chat correctly refused (409) after emptying the index.
- langchain-community deprecation: `langchain-faiss` on PyPI is an empty placeholder,
  so FAISS stays on langchain-community (pinned); swap point documented.

## Notes / risks

- Switching `PROVIDER` (or embedding model) switches the index dir automatically;
  previously ingested docs won't appear until re-ingested under the new model.
  Documented in README.
- FAISS `delete()` support depends on installed langchain/faiss versions — verified
  in venv before finalizing; fallback = rebuild index from uploads.
- `fake` provider is dev-only; answers are deterministic echoes, not real LLM output.
