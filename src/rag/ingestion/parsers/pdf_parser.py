"""PDF document parser extracting text with page provenance."""

import io
from typing import Any
from pypdf import PdfReader
from rag.domain.exceptions import DocumentParsingError
from rag.domain.models import Document, DocumentMetadata
from rag.ingestion.parsers.base import BaseParser


class PDFParser(BaseParser):
    """Parses PDF documents extracting text, page counts, and embedded metadata."""

    def parse(
        self,
        content: bytes | str,
        filename: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Document:
        if isinstance(content, str):
            content = content.encode("utf-8")

        content_hash = self.compute_sha256(content)

        try:
            reader = PdfReader(io.BytesIO(content))
            pages_text: list[str] = []
            page_metadata: list[dict[str, Any]] = []

            for page_idx, page in enumerate(reader.pages):
                extracted = page.extract_text() or ""
                pages_text.append(f"--- [Page {page_idx + 1}] ---\n{extracted}")
                page_metadata.append({
                    "page_number": page_idx + 1,
                    "character_count": len(extracted),
                })

            full_text = "\n\n".join(pages_text)

            # Extract PDF Document info if present
            pdf_info = reader.metadata or {}
            title = pdf_info.title or filename
            author = pdf_info.author or None

            metadata = DocumentMetadata(
                source_uri=filename,
                title=str(title),
                author=str(author) if author else None,
                content_hash=content_hash,
                extra={
                    **(extra_metadata or {}),
                    "page_count": len(reader.pages),
                    "page_details": page_metadata,
                    "format": "pdf",
                },
            )

            return Document(
                content=full_text,
                metadata=metadata,
            )
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse PDF {filename}: {e}") from e

