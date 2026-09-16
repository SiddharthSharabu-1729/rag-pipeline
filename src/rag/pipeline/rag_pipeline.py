"""End-to-end production RAG pipeline orchestrator with streaming and observability."""

import asyncio
import time
from typing import Any, AsyncIterator
from rag.config import settings
from rag.domain.models import (
    Citation,
    LatencyBreakdown,
    Query,
    RAGResponse,
    SearchResult,
)
from rag.generation.guardrails import Guardrails
from rag.generation.llm_client import BaseLLMClient, LLMFactory
from rag.generation.prompt_templates import (
    CORRECTION_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from rag.generation.self_rag_verifier import SelfRAGVerifier
from rag.observability.tracer import tracer
from rag.retrieval.context_optimizer import ContextOptimizer
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.reranker import BaseReranker, CrossEncoderReranker
from rag.storage.base import BaseEmbeddingClient, BaseVectorStore
from rag.storage.embeddings import EmbeddingFactory
from rag.storage.qdrant_store import QdrantVectorStore
from rag.storage.semantic_cache import SemanticCache


class RAGPipeline:
    """Production-grade RAG pipeline orchestrator."""

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        embedding_client: BaseEmbeddingClient | None = None,
        reranker: BaseReranker | None = None,
        llm_client: BaseLLMClient | None = None,
        semantic_cache: SemanticCache | None = None,
    ) -> None:
        self.embedding_client = embedding_client or EmbeddingFactory.get_client()
        self.vector_store = vector_store or QdrantVectorStore(dense_dim=self.embedding_client.dimension)
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_client=self.embedding_client,
        )
        self.reranker = reranker or CrossEncoderReranker()
        self.context_optimizer = ContextOptimizer()
        self.llm_client = llm_client or LLMFactory.get_client()
        self.verifier = SelfRAGVerifier()
        self.cache = semantic_cache or SemanticCache()

    async def initialize(self) -> None:
        """Initialize underlying vector store collections and indexes."""
        await self.vector_store.initialize()

    async def aquery(
        self,
        query_text: str,
        session_id: str | None = None,
        user_id: str | None = None,
        acl_groups: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> RAGResponse:
        """Execute complete synchronous-style RAG pipeline returning full RAGResponse."""
        t_start = time.perf_counter()
        latencies = LatencyBreakdown()
        trace = tracer.start_trace(query_text, metadata={"session_id": session_id, "user_id": user_id})

        # 0. Safety Guardrail
        with tracer.span(trace, "safety_guardrail"):
            safe, safety_msg = Guardrails.validate_safety(query_text)
            if not safe:
                await tracer.record_trace(trace)
                return RAGResponse(
                    query=query_text,
                    answer=f"Request blocked by safety guardrails: {safety_msg}",
                    status="insufficient_context",
                    trace_id=trace.trace_id,
                )

        query = Query(
            text=query_text,
            session_id=session_id,
            user_id=user_id,
            acl_groups=acl_groups or [],
            filters=filters or {},
        )

        # 1. Semantic Cache Probe
        t_cache = time.perf_counter()
        async with tracer.aspan(trace, "semantic_cache_probe"):
            query_vector = await self.embedding_client.aembed_query(query.text)
            cached_response = await self.cache.get(query_vector)
            latencies.cache_check_ms = (time.perf_counter() - t_cache) * 1000.0

            if cached_response:
                cached_response.latency.total_latency_ms = (time.perf_counter() - t_start) * 1000.0
                cached_response.trace_id = trace.trace_id
                await tracer.record_trace(trace)
                return cached_response

        # 2. Hybrid Retrieval (Dense + Sparse BM25 + Multi-Query)
        t_retrieve = time.perf_counter()
        async with tracer.aspan(trace, "hybrid_retrieval"):
            candidates = await self.retriever.retrieve(query, history=history)
            latencies.dense_retrieval_ms = (time.perf_counter() - t_retrieve) * 1000.0

        # 3. Cross-Encoder Re-ranking
        t_rerank = time.perf_counter()
        async with tracer.aspan(trace, "cross_rerank", attributes={"candidates_count": len(candidates)}):
            reranked = await self.reranker.arerank(query.text, candidates)
            latencies.fusion_rerank_ms = (time.perf_counter() - t_rerank) * 1000.0

        # 4. Quality Gate & Fallback Evaluation
        with tracer.span(trace, "quality_gate"):
            quality_passed = self.context_optimizer.evaluate_quality_gate(reranked)
        if not quality_passed:
            latencies.total_latency_ms = (time.perf_counter() - t_start) * 1000.0
            await tracer.record_trace(trace)
            return RAGResponse(
                query=query.text,
                answer="Based on the indexed documentation, I do not have sufficient information to answer your question.",
                citations=[],
                retrieved_chunks=reranked,
                latency=latencies,
                status="fallback",
                trace_id=trace.trace_id,
            )

        # 5. Lost-in-the-Middle Context Optimization & Citation Preparation
        with tracer.span(trace, "context_optimization"):
            optimized_chunks = self.context_optimizer.mitigate_lost_in_the_middle(reranked)
            context_str, citations = self.context_optimizer.assemble_context_and_citations(optimized_chunks)

        # 6. LLM Generation
        t_gen = time.perf_counter()
        async with tracer.aspan(trace, "llm_generation", attributes={"citations_count": len(citations)}):
            user_prompt = USER_PROMPT_TEMPLATE.format(context=context_str, query=query.text)
            draft_answer = await self.llm_client.agenerate(SYSTEM_PROMPT, user_prompt)
            latencies.generation_ms = (time.perf_counter() - t_gen) * 1000.0

        # 7. Self-RAG Verification & Correction
        t_verify = time.perf_counter()
        async with tracer.aspan(trace, "self_rag_verification"):
            faithfulness, ungrounded = self.verifier.verify_faithfulness(draft_answer, context_str)
            citations_valid = self.verifier.verify_citations(draft_answer, citations)

            final_answer = draft_answer
            if (faithfulness < 0.70 or not citations_valid) and settings.max_correction_retries > 0:
                correction_prompt = CORRECTION_PROMPT_TEMPLATE.format(
                    context=context_str,
                    query=query.text,
                    draft_response=draft_answer,
                )
                final_answer = await self.llm_client.agenerate(SYSTEM_PROMPT, correction_prompt)
                faithfulness, _ = self.verifier.verify_faithfulness(final_answer, context_str)

            latencies.self_rag_verification_ms = (time.perf_counter() - t_verify) * 1000.0

        # 8. Output PII Masking
        with tracer.span(trace, "pii_masking"):
            sanitized_answer = Guardrails.mask_pii(final_answer)

        latencies.total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        response = RAGResponse(
            query=query.text,
            answer=sanitized_answer,
            citations=citations,
            retrieved_chunks=optimized_chunks,
            latency=latencies,
            from_cache=False,
            faithfulness_score=faithfulness,
            model_name=getattr(self.llm_client, "model", "default"),
            status="success",
            trace_id=trace.trace_id,
        )

        # 9. Store in Semantic Cache
        await self.cache.put(query.text, query_vector, response)

        # Record completed trace
        await tracer.record_trace(trace)

        return response

    async def astream(
        self,
        query_text: str,
        acl_groups: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream generation tokens as Server-Sent Events (SSE) compatible dictionaries."""
        t_start = time.perf_counter()

        # Input Safety Check
        safe, msg = Guardrails.validate_safety(query_text)
        if not safe:
            yield {"event": "error", "data": f"Blocked by safety check: {msg}"}
            return

        query = Query(
            text=query_text,
            acl_groups=acl_groups or [],
            filters=filters or {},
        )

        # Retrieval & Re-ranking
        candidates = await self.retriever.retrieve(query, history=history)
        reranked = await self.reranker.arerank(query.text, candidates)

        if not self.context_optimizer.evaluate_quality_gate(reranked):
            yield {
                "event": "message",
                "data": "Based on the indexed documentation, I do not have sufficient information to answer your question.",
            }
            yield {"event": "done", "data": {"status": "fallback"}}
            return

        optimized_chunks = self.context_optimizer.mitigate_lost_in_the_middle(reranked)
        context_str, citations = self.context_optimizer.assemble_context_and_citations(optimized_chunks)

        # Stream Citations Header First
        citations_data = [c.model_dump() for c in citations]
        yield {"event": "citations", "data": citations_data}

        # Stream Generated Tokens
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context_str, query=query.text)
        ttft_recorded = False

        async for token in self.llm_client.astream(SYSTEM_PROMPT, user_prompt):
            if not ttft_recorded:
                ttft_ms = (time.perf_counter() - t_start) * 1000.0
                yield {"event": "ttft", "data": {"ttft_ms": round(ttft_ms, 2)}}
                ttft_recorded = True

            yield {"event": "token", "data": token}

        total_ms = (time.perf_counter() - t_start) * 1000.0
        yield {
            "event": "done",
            "data": {
                "total_latency_ms": round(total_ms, 2),
                "chunks_retrieved": len(reranked),
            },
        }

