"""Markdown parser with structural hierarchy extraction."""

import re
from typing import Any
from rag.domain.models import Document, DocumentMetadata
from rag.ingestion.parsers.base import BaseParser


class MarkdownParser(BaseParser):
    """Parses markdown content, extracting document title and hierarchy headers."""

    def parse(
        self,
        content: bytes | str,
        filename: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Document:
        text = content.decode("utf-8") if isinstance(content, bytes) else content
        content_hash = self.compute_sha256(text)

        # Extract title from first H1 if present
        title = filename
        h1_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()

        # Extract section headers
        headers = re.findall(r"^(#{1,4})\s+(.+)$", text, flags=re.MULTILINE)
        section_breadcrumbs = [f"{h[0]} {h[1].strip()}" for h in headers]

        metadata = DocumentMetadata(
            source_uri=filename,
            title=title,
            content_hash=content_hash,
            extra={
                **(extra_metadata or {}),
                "sections": section_breadcrumbs,
                "format": "markdown",
            },
        )

        return Document(
            content=text,
            metadata=metadata,
        )

