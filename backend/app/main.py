"""FastAPI application entrypoint."""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import chat, documents, health
from app.config import get_settings

logger = logging.getLogger("chatbot")


def create_app() -> FastAPI:
    settings = get_settings()

    # Ensure runtime directories exist.
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="RAG Document Chatbot API",
        version=__version__,
        description=(
            "Upload large documents (PDF, DOCX, TXT, MD); ask questions; "
            "get answers grounded in vector-searched context via LangChain."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {"service": "rag-document-chatbot", "version": __version__, "docs": "/docs"}

    return app


app = create_app()
