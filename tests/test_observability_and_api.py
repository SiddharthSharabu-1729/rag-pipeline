"""Tests for distributed tracing, observability spans, and API static routes."""

import pytest
from httpx import AsyncClient, ASGITransport
from rag.api.app import app
from rag.domain.models import Document, DocumentMetadata
from rag.observability.tracer import PipelineTracer, tracer
from rag.pipeline.rag_pipeline import RAGPipeline
from rag.storage.embeddings import MockEmbeddingClient
from rag.storage.qdrant_store import QdrantVectorStore


@pytest.mark.asyncio
async def test_tracer_spans_and_recording():
    custom_tracer = PipelineTracer(max_history=10)
    trace = custom_tracer.start_trace("How do containers work?")

    with custom_tracer.span(trace, "test_sync_span", attributes={"key": "val"}):
        pass

    async with custom_tracer.aspan(trace, "test_async_span"):
        pass

    await custom_tracer.record_trace(trace)
    traces = await custom_tracer.get_recent_traces(limit=5)

    assert len(traces) == 1
    t = traces[0]
    assert t["query"] == "How do containers work?"
    assert len(t["spans"]) == 2
    assert t["spans"][0]["name"] == "test_sync_span"
    assert t["spans"][0]["attributes"]["key"] == "val"
    assert t["spans"][1]["name"] == "test_async_span"


@pytest.mark.asyncio
async def test_pipeline_records_trace_id():
    dim = 64
    embedder = MockEmbeddingClient(dimension=dim)
    store = QdrantVectorStore(collection_name="test_obs_pipeline", dense_dim=dim)
    await store.initialize()
    pipeline = RAGPipeline(vector_store=store, embedding_client=embedder)

    resp = await pipeline.aquery("What is the system latency?")
    assert resp.trace_id is not None
    assert len(resp.trace_id) > 0


@pytest.mark.asyncio
async def test_api_service_and_traces_routes():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Test root discovery endpoint
        root_res = await client.get("/")
        assert root_res.status_code == 200
        data = root_res.json()
        assert data["docs_url"] == "/docs"
        assert data["health_url"] == "/api/v1/health"

        # Test health check endpoint
        health_res = await client.get("/api/v1/health")
        assert health_res.status_code == 200
        assert health_res.json()["status"] == "healthy"

        # Test traces route
        traces_res = await client.get("/api/v1/traces")
        assert traces_res.status_code == 200
        assert "traces" in traces_res.json()

