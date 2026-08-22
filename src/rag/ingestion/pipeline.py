"""Ingestion pipeline orchestrating parsing, hierarchical chunking, dual embedding, and upsert."""

from pathlib import Path
from typing import Any
from rag.domain.exceptions import DocumentParsingError, IngestionError
from rag.domain.models import Document, DocumentMetadata
from rag.ingestion.chunkers.base import BaseChunker
from rag.ingestion.chunkers.hierarchical import HierarchicalChunker
from rag.ingestion.parsers.factory import ParserFactory
from rag.storage.base import BaseEmbeddingClient, BaseVectorStore
from rag.storage.embeddings import EmbeddingFactory
from rag.storage.sparse import SparseBM25Encoder


class IngestionPipeline:
    """Production document ingestion ETL pipeline with Change Data Capture (CDC)."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_client: BaseEmbeddingClient | None = None,
        chunker: BaseChunker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_client = embedding_client or EmbeddingFactory.get_client()
        self.chunker = chunker or HierarchicalChunker()
        self.sparse_encoder = SparseBM25Encoder()
        # Track known document content hashes for local CDC deduplication
        self._known_doc_hashes: dict[str, str] = {}  # doc_id -> hash

    async def ingest_document(
        self,
        document: Document,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """Process and index a Document entity."""
        content_hash = document.metadata.content_hash

        # CDC Deduplication check
        if not force_reindex and self._known_doc_hashes.get(document.id) == content_hash:
            return {
                "doc_id": document.id,
                "title": document.metadata.title,
                "chunks_count": 0,
                "status": "skipped_duplicate",
                "message": "Document content identical to indexed version (CDC skip).",
            }

        # 1. Chunk document hierarchically
        chunks = self.chunker.chunk(document)
        if not chunks:
            return {
                "doc_id": document.id,
                "title": document.metadata.title,
                "chunks_count": 0,
                "status": "empty",
                "message": "Document produced 0 chunks.",
            }

        # 2. Extract texts for batch dense embedding
        chunk_texts = [c.content for c in chunks]
        dense_vectors = await self.embedding_client.aembed_documents(chunk_texts)

        # 3. Attach dense and sparse vectors to each chunk
        for chunk, d_vec in zip(chunks, dense_vectors):
            chunk.dense_vector = d_vec
            chunk.sparse_vector = self.sparse_encoder.encode(chunk.content)

        # 4. If re-indexing an existing document, delete previous chunks first
        if self._known_doc_hashes.get(document.id):
            await self.vector_store.delete_by_doc_id(document.id)

        # 5. Upsert to vector database
        upserted_count = await self.vector_store.upsert_chunks(chunks)

        # 6. Record hash for CDC
        self._known_doc_hashes[document.id] = content_hash

        return {
            "doc_id": document.id,
            "title": document.metadata.title,
            "chunks_count": upserted_count,
            "status": "indexed",
            "message": f"Successfully indexed {upserted_count} chunks.",
        }

    async def ingest_file(
        self,
        file_path: str | Path,
        extra_metadata: dict[str, Any] | None = None,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """Read and parse a file from disk and index it."""
        path = Path(file_path)
        if not path.is_file():
            raise DocumentParsingError(f"File not found: {file_path}")

        parser = ParserFactory.get_parser(path)
        content_bytes = path.read_bytes()
        document = parser.parse(
            content=content_bytes,
            filename=path.name,
            extra_metadata=extra_metadata,
        )
        return await self.ingest_document(document, force_reindex=force_reindex)

    async def ingest_text(
        self,
        text: str,
        title: str = "raw_text_document",
        extra_metadata: dict[str, Any] | None = None,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """Ingest raw in-memory string content."""
        parser = ParserFactory.get_parser(".txt")
        document = parser.parse(
            content=text,
            filename=title,
            extra_metadata=extra_metadata,
        )
        return await self.ingest_document(document, force_reindex=force_reindex)
