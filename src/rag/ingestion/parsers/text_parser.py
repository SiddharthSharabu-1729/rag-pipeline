"""Plain text document parser."""

from typing import Any
from rag.domain.models import Document, DocumentMetadata
from rag.ingestion.parsers.base import BaseParser


class TextParser(BaseParser):
    """Parses plain text files."""

    def parse(
        self,
        content: bytes | str,
        filename: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Document:
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        content_hash = self.compute_sha256(text)

        metadata = DocumentMetadata(
            source_uri=filename,
            title=filename,
            content_hash=content_hash,
            extra={
                **(extra_metadata or {}),
                "format": "text",
            },
        )

        return Document(
            content=text,
            metadata=metadata,
        )

