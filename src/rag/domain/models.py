"""Core domain entities and data transfer objects (DTOs) for the RAG pipeline."""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4
from pydantic import BaseModel, Field


class SparseVector(BaseModel):
    """Sparse vector representation for lexical search (e.g., BM25/SPLADE)."""

    indices: list[int] = Field(default_factory=list, description="Non-zero token/feature indices")
    values: list[float] = Field(default_factory=list, description="Corresponding weights/frequencies")


class DocumentMetadata(BaseModel):
    """Enriched metadata associated with an ingested source document."""

    source_uri: str = Field(description="File path, URL, or data store identifier")
    title: str = Field(default="", description="Document title or filename")
    author: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    acl_groups: list[str] = Field(default_factory=list, description="Tenant or role-based access control tags")
    content_hash: str = Field(default="", description="SHA-256 hash of raw content for CDC deduplication")
    extra: dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra metadata")


class Document(BaseModel):
    """Raw parsed document before chunking."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    metadata: DocumentMetadata


class Chunk(BaseModel):
    """A granular chunk of a document, supporting hierarchical parent-child relationships."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str = Field(description="ID of the parent Document")
    parent_id: str | None = Field(default=None, description="ID of the parent contextual chunk (if hierarchical)")
    parent_content: str | None = Field(default=None, description="Expanded parent section text for synthesis")
    content: str = Field(description="Granular leaf chunk text used for embedding similarity")
    chunk_type: Literal["text", "table", "code", "header"] = "text"
    page_number: int | None = None
    token_count: int = 0
    dense_vector: list[float] | None = None
    sparse_vector: SparseVector | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Query(BaseModel):
    """Runtime search query parameters."""

    text: str
    session_id: str | None = None
    user_id: str | None = None
    acl_groups: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int | None = None


class SearchResult(BaseModel):
    """A single retrieved chunk match with scoring and provenance."""

    chunk: Chunk
    score: float = Field(description="Unified or reranked relevance score")
    retrieval_method: Literal["dense", "sparse", "hybrid", "reranked"] = "hybrid"
    parent_content: str | None = None


class Citation(BaseModel):
    """Grounding citation referencing the source chunk and document."""

    source_id: str = Field(description="Document or chunk ID")
    title: str
    source_uri: str
    page_number: int | None = None
    snippet: str = Field(description="Relevant text excerpt supporting the statement")


class LatencyBreakdown(BaseModel):
    """Fine-grained latency tracking for observability."""

    cache_check_ms: float = 0.0
    query_transform_ms: float = 0.0
    dense_retrieval_ms: float = 0.0
    sparse_retrieval_ms: float = 0.0
    fusion_rerank_ms: float = 0.0
    generation_ms: float = 0.0
    self_rag_verification_ms: float = 0.0
    total_latency_ms: float = 0.0


class RAGResponse(BaseModel):
    """Complete structured response from the RAG generation pipeline."""

    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[SearchResult] = Field(default_factory=list)
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    from_cache: bool = False
    faithfulness_score: float | None = None
    model_name: str = ""
    status: Literal["success", "fallback", "insufficient_context"] = "success"
    trace_id: str | None = None


class EvaluationMetrics(BaseModel):
    """RAG Triad and operational quality metrics."""

    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)

