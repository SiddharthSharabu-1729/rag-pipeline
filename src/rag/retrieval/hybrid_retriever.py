"""Hybrid retrieval engine orchestrating multi-query expansion and RRF fusion."""

import asyncio
from typing import Any
from rag.config import settings
from rag.domain.models import Query, SearchResult
from rag.retrieval.query_transform import QueryTransformer
from rag.storage.base import BaseEmbeddingClient, BaseVectorStore


class HybridRetriever:
    """Executes multi-query expansion, dense search, and sparse BM25 search with RRF."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_client: BaseEmbeddingClient,
        top_k: int = settings.top_k_retrieval,
        rrf_k: int = settings.rrf_k,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.transformer = QueryTransformer()

    async def retrieve(
        self,
        query: Query,
        history: list[dict[str, str]] | None = None,
    ) -> list[SearchResult]:
        """Perform end-to-end multi-stage hybrid retrieval."""
        # 1. Sanitize & Contextualize
        clean_text = self.transformer.sanitize(query.text)
        contextualized = self.transformer.contextualize_query(clean_text, history)
        extracted_filters = self.transformer.extract_filters(clean_text)

        # Merge extracted filters with existing query filters
        merged_filters = {**extracted_filters, **query.filters}
        effective_query = query.model_copy(
            update={"text": contextualized, "filters": merged_filters}
        )

        # 2. Sub-query decomposition
        sub_queries = self.transformer.decompose_query(contextualized)

        # 3. Parallel retrieval across sub-queries
        tasks = []
        for sq in sub_queries:
            sub_q_obj = effective_query.model_copy(update={"text": sq})
            tasks.append(self._retrieve_single(sub_q_obj))

        results_by_subquery = await asyncio.gather(*tasks)

        # 4. Merge and deduplicate candidates across sub-queries
        merged_candidates: dict[str, SearchResult] = {}
        for sub_results in results_by_subquery:
            for res in sub_results:
                cid = res.chunk.chunk_id
                if cid not in merged_candidates or res.score > merged_candidates[cid].score:
                    merged_candidates[cid] = res

        # Sort by score descending
        sorted_results = sorted(
            merged_candidates.values(),
            key=lambda r: r.score,
            reverse=True,
        )

        return sorted_results[: self.top_k]

    async def _retrieve_single(self, query: Query) -> list[SearchResult]:
        """Retrieve candidates for an individual query."""
        # Compute dense query vector
        query_vector = await self.embedding_client.aembed_query(query.text)

        # Hybrid search combining dense + sparse BM25
        results = await self.vector_store.search_hybrid(
            query=query,
            query_dense_vector=query_vector,
            limit=self.top_k,
            rrf_k=self.rrf_k,
        )
        return results

