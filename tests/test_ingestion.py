"""Tests for document parsers and hierarchical chunkers."""

import pytest
from rag.domain.models import Document, DocumentMetadata
from rag.ingestion.chunkers.hierarchical import HierarchicalChunker
from rag.ingestion.parsers.factory import ParserFactory
from rag.ingestion.parsers.markdown_parser import MarkdownParser
from rag.ingestion.parsers.text_parser import TextParser


def test_markdown_parser_extracts_title_and_sections():
    parser = MarkdownParser()
    md_content = """# Architecture Overview

This is the system architecture documentation.

## Storage Layer
Details about storage.

### Qdrant Vector Store
Details about vector indexing.
"""
    doc = parser.parse(md_content, filename="architecture.md")

    assert doc.metadata.title == "Architecture Overview"
    assert doc.metadata.source_uri == "architecture.md"
    assert len(doc.metadata.content_hash) == 64
    assert len(doc.metadata.extra.get("sections", [])) >= 2


def test_text_parser():
    parser = TextParser()
    doc = parser.parse("Simple plain text content.", filename="log.txt")
    assert doc.metadata.title == "log.txt"
    assert doc.content == "Simple plain text content."


def test_parser_factory_resolution():
    p1 = ParserFactory.get_parser("doc.md")
    p2 = ParserFactory.get_parser("doc.pdf")
    p3 = ParserFactory.get_parser("doc.txt")

    assert isinstance(p1, MarkdownParser)
    assert p2.__class__.__name__ == "PDFParser"
    assert isinstance(p3, TextParser)


def test_hierarchical_chunker_parent_child_linking():
    chunker = HierarchicalChunker(
        parent_chunk_size=100,
        parent_chunk_overlap=20,
        leaf_chunk_size=30,
        leaf_chunk_overlap=5,
    )

    doc_text = (
        "Production RAG requires robust data ingestion. " * 15 + "\n\n"
        "Hybrid search combines dense embeddings and BM25 lexical search. " * 15 + "\n\n"
        "Cross-encoders eliminate false positives and score precision. " * 15
    )

    doc = Document(
        id="test-doc-001",
        content=doc_text,
        metadata=DocumentMetadata(source_uri="test.txt", title="Test Doc"),
    )

    chunks = chunker.chunk(doc)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.doc_id == "test-doc-001"
        assert chunk.parent_id is not None
        assert chunk.parent_content is not None
        # Parent context should contain leaf sentences
        first_sentence = chunk.content.split(".")[0].strip()
        assert first_sentence in chunk.parent_content
        assert chunk.token_count > 0
