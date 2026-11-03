"""Application configuration, loaded from environment variables / .env file."""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ folder (this file lives at backend/app/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings. Every value can be overridden via .env or environment."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Provider selection -------------------------------------------------
    # openai  -> OpenAI (or any OpenAI-compatible endpoint via OPENAI_BASE_URL)
    # ollama  -> local Ollama server (no API key required)
    # mistral -> Mistral AI API (MISTRAL_API_KEY)
    # fake    -> fully offline deterministic stubs (dev / smoke tests only)
    provider: Literal["openai", "ollama", "mistral", "fake"] = "openai"

    # --- OpenAI --------------------------------------------------------------
    openai_api_key: str | None = None
    openai_base_url: str | None = None  # set for OpenAI-compatible endpoints
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Ollama --------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1"
    ollama_embedding_model: str = "nomic-embed-text"

    # --- Mistral AI -------------------------------------------------------------
    mistral_api_key: str | None = None
    mistral_base_url: str | None = None  # defaults to https://api.mistral.ai/v1/ when unset
    mistral_chat_model: str = "mistral-small"
    mistral_embedding_model: str = "mistral-embed"

    # --- Chunking / retrieval --------------------------------------------------
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    top_k: int = Field(default=4, gt=0, le=20)

    # --- Storage ---------------------------------------------------------------
    upload_dir: Path = BASE_DIR / "storage" / "uploads"
    vectorstore_dir: Path = BASE_DIR / "storage" / "vectorstore"

    # --- Uploads -----------------------------------------------------------------
    max_upload_size_mb: int = Field(default=50, gt=0)

    # --- CORS (comma separated, e.g. "http://localhost:3000,http://127.0.0.1:5173")
    cors_origins: str = "*"

    @model_validator(mode="after")
    def overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than CHUNK_SIZE ({self.chunk_size})"
            )
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def chat_model(self) -> str:
        return {
            "openai": self.openai_chat_model,
            "ollama": self.ollama_chat_model,
            "mistral": self.mistral_chat_model,
            "fake": "fake-echo",
        }[self.provider]

    @property
    def embedding_model(self) -> str:
        return {
            "openai": self.openai_embedding_model,
            "ollama": self.ollama_embedding_model,
            "mistral": self.mistral_embedding_model,
            "fake": "fake-embedding",
        }[self.provider]

    @property
    def embedding_slug(self) -> str:
        """Filesystem-safe identifier for the current embedding model.

        The FAISS index is stored per embedding model so switching providers
        never mixes incompatible vectors.
        """
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.embedding_model).strip("_")
        return slug or "unknown"

    @property
    def index_dir(self) -> Path:
        return self.vectorstore_dir / self.embedding_slug


@lru_cache
def get_settings() -> Settings:
    return Settings()
