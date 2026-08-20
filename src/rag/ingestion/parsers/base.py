"""Base parser interface for document ingestion."""

from abc import ABC, abstractmethod
import hashlib
from typing import Any
from rag.domain.models import Document, DocumentMetadata


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(
        self,
        content: bytes | str,
        filename: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Parse raw document bytes or string into a standardized Document entity."""
        pass

    @staticmethod
    def compute_sha256(content: bytes | str) -> str:
        """Compute SHA-256 hash of document content for deduplication and CDC."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

