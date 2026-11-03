"""Application configuration, loaded from environment variables / .env file.

The stack is deliberately small: **Gemini** for chat, **Pinecone** for both
embeddings (hosted inference) and vector search. `PROVIDER=fake` is a
tests/offline mode that swaps in deterministic stubs.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VALID_PROVIDERS = ("gemini", "fake")

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
    # gemini -> Google Gemini for the LLM, Pinecone for embeddings + vectors
    # fake   -> fully offline deterministic stubs (tests / smoke checks only;
    #           still needs a Pinecone index unless the client is mocked)
    provider: Literal["gemini", "fake"] = "gemini"

    # --- Google Gemini (LLM) ------------------------------------------------
    google_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY")
    )
    google_base_url: str | None = Field(
        default=None, validation_alias=AliasChoices("GOOGLE_BASE_URL", "GEMINI_BASE_URL")
    )  # set only for proxies/gateways; default is Google's API endpoint
    google_chat_model: str = Field(
        default="gemini-2.5-flash", validation_alias=AliasChoices("GOOGLE_CHAT_MODEL", "GEMINI_CHAT_MODEL")
    )

    # --- Pinecone (embeddings + vector store) --------------------------------
    pinecone_api_key: str | None = None
    pinecone_index: str | None = None  # index name; must already exist (dimension must match the embedding model)
    pinecone_host: str | None = None  # alternative to the name: full index host URL (paste from the console)
    pinecone_namespace: str = ""  # "" = Pinecone's default namespace
    # Hosted embedding model used through Pinecone's inference API (1024-dim dense
    # models: llama-text-embed-v2, multilingual-e5-large).
    pinecone_embedding_model: str = Field(
        default="llama-text-embed-v2",
        validation_alias=AliasChoices("PINECONE_EMBEDDING_MODEL", "PINECONE_EMBED_MODEL"),
    )

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

    @field_validator("provider", mode="before")
    @classmethod
    def blank_provider_means_default(cls, value):
        """Compose passes `PROVIDER=${PROVIDER:-gemini}`; also tolerate a blank .env value."""
        if isinstance(value, str) and not value.strip():
            return "gemini"
        return value

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
        return "fake-echo" if self.provider == "fake" else self.google_chat_model

    @property
    def embedding_model(self) -> str:
        return "fake-embedding" if self.provider == "fake" else self.pinecone_embedding_model

    @property
    def embedding_slug(self) -> str:
        """Filesystem-safe identifier for the current embedding model.

        The document manifest is stored per embedding model so switching models
        never mixes incompatible bookkeeping.
        """
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.embedding_model).strip("_")
        return slug or "unknown"

    @property
    def index_dir(self) -> Path:
        return self.vectorstore_dir / self.embedding_slug


@lru_cache
def get_settings() -> Settings:
    return Settings()
