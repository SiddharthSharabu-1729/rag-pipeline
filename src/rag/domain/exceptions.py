"""Domain-specific exceptions for the RAG pipeline."""


class RAGException(Exception):
    """Base exception for all RAG pipeline errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DocumentParsingError(RAGException):
    """Raised when parsing a source document fails."""
    pass


class IngestionError(RAGException):
    """Raised when document ingestion fails."""
    pass


class StorageError(RAGException):
    """Raised when an error occurs during vector/payload storage operations."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding computation fails."""
    pass


class RetrievalError(RAGException):
    """Raised when query retrieval or reranking fails."""
    pass


class GenerationError(RAGException):
    """Raised when LLM text generation fails."""
    pass


class FallbackTriggered(RAGException):
    """Raised or returned when retrieved context fails quality thresholds."""
    pass
