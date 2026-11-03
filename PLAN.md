# PLAN.md — RAG Chatbot Backend

Working plan for the document-Q&A chatbot backend. Kept up to date as tasks land.

## Goal

Build a production-shaped FastAPI backend that:

1. Accepts large documents (PDF / DOCX / TXT / MD) via upload.
2. Parses → chunks → embeds → stores them in **Pinecone**.
3. On each query: embeds the query → vector search → stuffs retrieved chunks
   into a LangChain prompt → calls **Gemini** → returns the answer **with source
   citations**.
4. Runs offline for tests via a `fake` provider (deterministic stubs + mocked
   Pinecone client).

Frontend is out of scope (user will build it later). Backend lives in `backend/`.

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Framework | FastAPI + uvicorn | Requirement; async, OpenAPI docs out of the box |
| RAG orchestration | LangChain | Requirement |
| LLM | `langchain-google-genai` (`gemini-2.5-flash` default, configurable) | Single provider per user's final request |
| Embeddings | Pinecone hosted inference (`pc.inference.embed`) — `llama-text-embed-v2` (1024-dim) default | User wanted Pinecone to handle embeddings; no separate embedding service |
| Vector store | Pinecone (serverless index, cosine) via the official `pinecone` SDK | Managed vector search, deletion by id |
| Pinecone LangChain adapter | Own thin `VectorStore` subclass (`app/services/pinecone_store.py`) | `langchain-pinecone` on PyPI is an abandoned 0.0.1 stub (numpy<2 → unbuildable on 3.14); `langchain-community` dropped Pinecone |
| Chunking | `RecursiveCharacterTextSplitter` (1000/200, configurable) | Solid default for prose/docs |
| Doc tracking | Local `manifest.json` per embedding model | List/delete without scanning index internals |
| Delete semantics | Pinecone `delete(ids)` with `{doc_id}:{chunk_index}` ids | Real removal, not just hiding from the manifest |
| Chat history | Stateless; client sends optional `history` array | No server-side session state, frontend owns state |
| API prefix | `/api` | Clean frontend integration |
| Provider switch | `PROVIDER=gemini\|fake` | `fake` = offline tests/smoke checks only |
| Containerization | Single `backend` service in `compose.yaml` | Gemini + Pinecone are external managed services; no local model/DB containers |
| Backend image | python:3.14-slim, non-root user, `/data` volume | Wheels only, no compilers; secrets never in image |

## Project layout

```
backend/
  app/
    __init__.py
    main.py            # app factory, routers, exception handlers
    config.py          # pydantic-settings (.env)
    schemas.py         # Pydantic request/response models
    api/routes/
      health.py        # GET /api/health
      documents.py     # POST /api/documents/upload, GET /api/documents, DELETE /api/documents/{id}
      chat.py          # POST /api/chat  (RAG: retrieve -> generate)
    services/
      document_loader.py   # extension dispatch -> LangChain loaders
      chunking.py          # RecursiveCharacterTextSplitter wrapper
      embeddings.py        # Pinecone hosted-inference embeddings
      pinecone_client.py   # client/index helpers with clear config errors
      pinecone_store.py    # thin LangChain VectorStore over the pinecone SDK
      vectorstore.py       # add/search/delete + document manifest
      llm.py               # Gemini factory (+ offline EchoChatModel for `fake`)
      rag.py               # retrieve -> prompt -> LLM -> answer+sources
  tests/                 # pytest, no network/keys required (fake Pinecone + fake LLM)
  requirements.txt       # pinned
  .env.example
  pytest.ini
README.md                # repo root, user-facing docs
PLAN.md                  # this file
compose.yaml             # backend service only
```

## API surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness + active models + docs indexed |
| POST | `/api/documents/upload` | Multipart upload (pdf/docx/txt/md) → ingest |
| GET | `/api/documents` | List ingested documents |
| DELETE | `/api/documents/{doc_id}` | Remove document chunks from Pinecone |
| POST | `/api/chat` | RAG query → `{answer, sources[]}` |

## Work checklist

- [x] Scaffold backend package + config + schemas
- [x] Ingestion: loader → chunker → vector store (+manifest, delete)
- [x] RAG: provider factories, retrieval chain, /api/chat
- [x] API routes + error handling + CORS + upload validation (ext/size)
- [x] requirements.txt (pinned), .env.example, .gitignore
- [x] Offline pytest suite (fake provider + fake Pinecone)
- [x] Dockerfile (slim, non-root, healthcheck, /data volume) + backend/.dockerignore
- [x] compose.yaml (backend only) + `docker compose config` validation
- [x] Gemini LLM (`langchain-google-genai`)
- [x] Pinecone for embeddings (hosted inference) + vector search (thin VectorStore adapter)
- [x] Live local-stack verification: upload → embed → search → Gemini/llama answer with sources
- [ ] Live Gemini + Pinecone verification (needs real `GOOGLE_API_KEY` + `PINECONE_API_KEY`)
- [x] Final container rebuild + health/upload/chat smoke test
- [x] README.md and PLAN.md updated for the final stack

## Verification summary

- `pytest` — 29/29 green, fully offline (fake LLM + in-memory fake Pinecone).
- `docker compose config` — valid.
- Container smoke test — boots healthy; health reports the Gemini/Pinecone
  configuration; a misconfigured upload returns a clear `503` with the missing
  setting named.
- Live Gemini + Pinecone calls remain unverified here (no real API keys
  available); config errors, embedding/upsert/query/delete semantics and the
  RAG pipeline are covered by offline tests.

## Notes / risks

- Pinecone index **dimension is fixed at creation** and must match the embedding
  model (1024 for the supported dense models). Documented in README.
- `manifest.json` is node-local: multiple backend replicas against one index each
  see only their own document list.
- Pinecone hosted inference has request quotas; ingestion embeds in batches.
- `PROVIDER=fake` is dev-only; answers are deterministic echoes, embeddings are
  fake vectors.
