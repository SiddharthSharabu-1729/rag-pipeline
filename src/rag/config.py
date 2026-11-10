"""Configuration settings for the Production RAG pipeline.

Uses Pydantic Settings for type-safe environment variable management and validation.
"""

from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration parameters with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General System
    app_name: str = "Production RAG Pipeline"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Vector Storage (Qdrant)
    qdrant_url: str | None = None  # None indicates in-memory or local path
    qdrant_api_key: str | None = None
    qdrant_path: str | None = None  # Local persistence directory if not in-memory
    qdrant_collection_name: str = "rag_enterprise_docs"
    use_in_memory_qdrant: bool = True  # Defaults to True for zero-friction local run/testing
    max_upload_bytes: int = 10 * 1024 * 1024

    # Embeddings
    embedding_provider: Literal["openai", "mock", "custom"] = "mock"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 384  # Standard dense embedding dimension
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = None

    # LLM Settings
    llm_provider: Literal["openai", "mock", "custom"] = "mock"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1500

    # Ingestion & Chunking
    parent_chunk_size: int = 800  # Token budget for parent context
    parent_chunk_overlap: int = 100
    leaf_chunk_size: int = 200    # Token budget for dense retrieval leaf chunks
    leaf_chunk_overlap: int = 40

    # Retrieval & Re-ranking
    top_k_retrieval: int = 20      # Stage 1: Broad candidates (dense + sparse)
    top_k_rerank: int = 5          # Stage 2: Top candidates fed into context
    rrf_k: int = 60                # Reciprocal Rank Fusion constant
    relevance_threshold: float = 0.35  # Minimum rerank score before fallback triggers
    reranker_provider: Literal["cross_encoder", "flashrank", "rule_based"] = "rule_based"

    # Semantic Caching
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.94  # Cosine similarity threshold for cache hit
    semantic_cache_max_entries: int = 1000

    # Guardrails & Verification (Self-RAG)
    enable_pii_masking: bool = True
    enable_self_rag_verification: bool = True
    max_correction_retries: int = 1


# Global singleton settings instance
settings = Settings()

