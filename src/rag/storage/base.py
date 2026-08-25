"""Abstract base interfaces for embeddings and vector storage."""

from abc import ABC, abstractmethod
from typing import Any
from rag.domain.models import Chunk, Query, SearchResult


class BaseEmbeddingClient(ABC):
    """Abstract interface for dense embedding generation."""

    @abstractmethod
    async def aembed_query(self, text: str) -> list[float]:
        """Generate a dense vector for a single query."""
        pass

    @abstractmethod
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate dense vectors for a batch of documents/chunks."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension."""
        pass


class BaseVectorStore(ABC):
    """Abstract interface for hybrid vector database."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create collections, indexes, and schemas."""
        pass

    @abstractmethod
    async def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """Store or update chunks with dense vectors, sparse vectors, and metadata payloads."""
        pass

    @abstractmethod
    async def search_dense(self, query_vector: list[float], query: Query, limit: int = 10) -> list[SearchResult]:
        """Perform dense semantic similarity search."""
        pass

    @abstractmethod
    async def search_sparse(self, query: Query, limit: int = 10) -> list[SearchResult]:
        """Perform lexical sparse (BM25) search."""
        pass

    @abstractmethod
    async def search_hybrid(
        self,
        query: Query,
        query_dense_vector: list[float],
        limit: int = 10,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Perform hybrid search combining dense + sparse with Reciprocal Rank Fusion."""
        pass

    @abstractmethod
    async def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document (for CDC updates)."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Return total number of indexed chunks."""
        pass

