"""RAG chat endpoint: vector search -> LLM -> grounded answer with sources."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.schemas import ChatRequest, ChatResponse
from app.services.rag import answer_query
from app.services.vectorstore import (
    VectorStoreNotReadyError,
    get_vectorstore_manager,
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    manager = get_vectorstore_manager()
    try:
        return await run_in_threadpool(
            lambda: answer_query(
                query=request.query,
                settings=settings,
                store=manager,
                top_k=request.top_k,
                history=request.history,
            )
        )
    except VectorStoreNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Missing API key / provider misconfiguration
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM or retrieval call failed: {exc}") from exc
