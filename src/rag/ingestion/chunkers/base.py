"""Base chunker interface and tokenization utilities."""

from abc import ABC, abstractmethod
import re
import tiktoken
from rag.domain.models import Chunk, Document


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    def __init__(self) -> None:
        # Fast local token estimation to guarantee zero network latency & zero blocking
        self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Estimate token count deterministically without network dependencies."""
        if not text:
            return 0
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return max(1, int(len(tokens) * 1.25))

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into a sequence of Chunk entities."""
        pass
