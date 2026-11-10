"""Qdrant hybrid vector store supporting dense and sparse BM25 indexing."""

import asyncio
from typing import Any
from uuid import UUID, uuid5, NAMESPACE_DNS
from qdrant_client import AsyncQdrantClient, models
from rag.config import settings
from rag.domain.exceptions import StorageError
from rag.domain.models import Chunk, Query, SearchResult, SparseVector
from rag.storage.base import BaseVectorStore
from rag.storage.sparse import SparseBM25Encoder


class QdrantVectorStore(BaseVectorStore):
    """Production vector store implementation using Qdrant with hybrid search."""

    def __init__(
        self,
        collection_name: str = settings.qdrant_collection_name,
        dense_dim: int = settings.embedding_dim,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.dense_dim = dense_dim
        self.sparse_encoder = SparseBM25Encoder()

        if client:
            self.client = client
        elif settings.use_in_memory_qdrant:
            self.client = AsyncQdrantClient(location=":memory:")
        elif settings.qdrant_url:
            self.client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        elif settings.qdrant_path:
            self.client = AsyncQdrantClient(path=settings.qdrant_path)
        else:
            self.client = AsyncQdrantClient(location=":memory:")

    async def initialize(self) -> None:
        """Create hybrid collection and payload indexes if not present."""
        try:
            exists = await self.client.collection_exists(self.collection_name)
            if not exists:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self.dense_dim,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams()
                    },
                )
                # Create payload indexes for metadata filtering
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="doc_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="parent_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="acl_groups",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
        except Exception as e:
            raise StorageError(f"Failed to initialize Qdrant collection: {e}") from e

    def _to_uuid_str(self, key: str) -> str:
        """Ensure point ID is a valid RFC 4122 UUID string."""
        try:
            return str(UUID(key))
        except ValueError:
            return str(uuid5(NAMESPACE_DNS, key))

    async def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """Upsert a list of chunks into Qdrant with dense and sparse vectors."""
        if not chunks:
            return 0

        points: list[models.PointStruct] = []
        for chunk in chunks:
            if not chunk.dense_vector:
                raise StorageError(f"Chunk {chunk.chunk_id} missing dense vector for upsert")

            # Compute sparse vector if not present
            sparse_vec = chunk.sparse_vector or self.sparse_encoder.encode(chunk.content)

            point_id = self._to_uuid_str(chunk.chunk_id)

            vectors: dict[str, Any] = {
                "dense": chunk.dense_vector,
            }
            if sparse_vec.indices:
                vectors["sparse"] = models.SparseVector(
                    indices=sparse_vec.indices,
                    values=sparse_vec.values,
                )

            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "parent_id": chunk.parent_id,
                "parent_content": chunk.parent_content,
                "content": chunk.content,
                "chunk_type": chunk.chunk_type,
                "page_number": chunk.page_number,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata,
                "acl_groups": chunk.metadata.get("acl_groups", []),
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload=payload,
                )
            )

        try:
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            return len(points)
        except Exception as e:
            raise StorageError(f"Failed to upsert chunks into Qdrant: {e}") from e

    def _build_filter(self, query: Query) -> models.Filter | None:
        """Build Qdrant Filter from query metadata and ACL groups."""
        must_conditions: list[models.Condition] = []

        acl_conditions: list[models.Condition] = []
        if query.acl_groups:
            acl_conditions.append(
                models.FieldCondition(
                    key="acl_groups",
                    match=models.MatchAny(any=query.acl_groups),
                )
            )
        # Empty or missing ACL metadata denotes a public document. With no
        # caller groups, only public documents are eligible for retrieval.
        acl_conditions.append(
            models.IsEmptyCondition(is_empty=models.PayloadField(key="acl_groups"))
        )

        for key, val in query.filters.items():
            if isinstance(val, list):
                must_conditions.append(
                    models.FieldCondition(key=f"metadata.{key}", match=models.MatchAny(any=val))
                )
            else:
                must_conditions.append(
                    models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=val))
                )

        return models.Filter(must=must_conditions, should=acl_conditions)

    def _point_to_search_result(
        self,
        point: models.ScoredPoint,
        retrieval_method: str = "dense",
    ) -> SearchResult:
        """Convert a ScoredPoint into domain SearchResult."""
        payload = point.payload or {}
        chunk = Chunk(
            chunk_id=payload.get("chunk_id", str(point.id)),
            doc_id=payload.get("doc_id", ""),
            parent_id=payload.get("parent_id"),
            parent_content=payload.get("parent_content"),
            content=payload.get("content", ""),
            chunk_type=payload.get("chunk_type", "text"),
            page_number=payload.get("page_number"),
            token_count=payload.get("token_count", 0),
            metadata=payload.get("metadata", {}),
        )
        return SearchResult(
            chunk=chunk,
            score=float(point.score),
            retrieval_method=retrieval_method,  # type: ignore[arg-type]
            parent_content=chunk.parent_content,
        )

    async def search_dense(
        self,
        query_vector: list[float],
        query: Query,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Execute dense cosine similarity search."""
        q_filter = self._build_filter(query)
        try:
            res = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using="dense",
                query_filter=q_filter,
                limit=limit,
                with_payload=True,
            )
            return [self._point_to_search_result(pt, retrieval_method="dense") for pt in res.points]
        except Exception as e:
            raise StorageError(f"Dense vector search failed: {e}") from e

    async def search_sparse(
        self,
        query: Query,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Execute sparse lexical (BM25) search."""
        sparse_vec = self.sparse_encoder.encode(query.text)
        if not sparse_vec.indices:
            return []

        q_filter = self._build_filter(query)
        try:
            res = await self.client.query_points(
                collection_name=self.collection_name,
                query=models.SparseVector(
                    indices=sparse_vec.indices,
                    values=sparse_vec.values,
                ),
                using="sparse",
                query_filter=q_filter,
                limit=limit,
                with_payload=True,
            )
            return [self._point_to_search_result(pt, retrieval_method="sparse") for pt in res.points]
        except Exception as e:
            raise StorageError(f"Sparse lexical search failed: {e}") from e

    async def search_hybrid(
        self,
        query: Query,
        query_dense_vector: list[float],
        limit: int = 10,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Combine dense and sparse search using Reciprocal Rank Fusion (RRF)."""
        # Execute dense and sparse in parallel
        dense_results, sparse_results = await asyncio.gather(
            self.search_dense(query_dense_vector, query, limit=limit * 2),
            self.search_sparse(query, limit=limit * 2),
        )

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, SearchResult] = {}

        # Accumulate RRF for dense
        for rank, res in enumerate(dense_results):
            cid = res.chunk.chunk_id
            chunk_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Accumulate RRF for sparse
        for rank, res in enumerate(sparse_results):
            cid = res.chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort by fused RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        hybrid_results: list[SearchResult] = []
        for cid in sorted_cids[:limit]:
            base_res = chunk_map[cid]
            hybrid_results.append(
                SearchResult(
                    chunk=base_res.chunk,
                    score=round(rrf_scores[cid], 5),
                    retrieval_method="hybrid",
                    parent_content=base_res.parent_content,
                )
            )

        return hybrid_results

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks for a given document ID."""
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=doc_id),
                            )
                        ]
                    )
                ),
            )
            return 1
        except Exception as e:
            raise StorageError(f"Failed to delete document {doc_id}: {e}") from e

    async def count(self) -> int:
        """Count total chunks in collection."""
        try:
            res = await self.client.count(collection_name=self.collection_name)
            return res.count
        except Exception:
            return 0

