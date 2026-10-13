"""End-to-end integration tests for the production RAG pipeline."""

import pytest
import pytest_asyncio
from rag.domain.models import Document, DocumentMetadata
from rag.evaluation.evaluator import BenchmarkEvaluator
from rag.ingestion.pipeline import IngestionPipeline
from rag.pipeline.rag_pipeline import RAGPipeline
from rag.storage.embeddings import MockEmbeddingClient
from rag.storage.qdrant_store import QdrantVectorStore


@pytest_asyncio.fixture
async def setup_pipeline():
    """Fixture providing initialized RAG and Ingestion pipelines."""
    dense_dim = 64
    embedder = MockEmbeddingClient(dimension=dense_dim)
    store = QdrantVectorStore(collection_name="test_e2e_rag", dense_dim=dense_dim)
    await store.initialize()

    pipeline = RAGPipeline(
        vector_store=store,
        embedding_client=embedder,
    )
    ingestion = IngestionPipeline(
        vector_store=store,
        embedding_client=embedder,
    )

    # Ingest sample enterprise document
    knowledge_base = """# Cloud Native Storage Architecture

## High Availability and Replication
The storage cluster runs with a 3-node Raft consensus group.
Data is synchronously replicated across three Availability Zones.
The cluster supports up to 100,000 IOPS per volume with sub-millisecond p99 latency.

## Disaster Recovery
Snapshots are automatically taken every 15 minutes and replicated to an S3-compatible cold tier.
Recovery Point Objective (RPO) is guaranteed at 15 minutes, and Recovery Time Objective (RTO) is under 2 minutes.

## Security and Encryption
All storage volumes are encrypted at rest using AES-256-GCM.
In-transit traffic is secured with mutual TLS (mTLS) 1.3.
Access control is enforced via Role-Based Access Control (RBAC) tokens.
"""
    doc = Document(
        id="cloud-storage-arch-v1",
        content=knowledge_base,
        metadata=DocumentMetadata(
            source_uri="docs/storage_architecture.md",
            title="Cloud Native Storage Architecture",
        ),
    )
    await ingestion.ingest_document(doc)

    return pipeline, ingestion


@pytest.mark.asyncio
async def test_end_to_end_query_and_grounding(setup_pipeline):
    pipeline, _ = setup_pipeline

    query = "What is the RPO and RTO for disaster recovery?"
    response = await pipeline.aquery(query)

    assert response.status == "success"
    assert len(response.citations) > 0
    assert "[Ref-" in response.answer
    assert response.from_cache is False
    assert response.latency.total_latency_ms > 0

    # Test Semantic Cache Hit on repeat query
    cached_response = await pipeline.aquery(query)
    assert cached_response.from_cache is True
    assert cached_response.answer == response.answer


@pytest.mark.asyncio
async def test_end_to_end_streaming(setup_pipeline):
    pipeline, _ = setup_pipeline

    query = "How is data encrypted and secured in-transit?"
    events = []
    async for event in pipeline.astream(query):
        events.append(event)

    event_names = [e["event"] for e in events]
    assert "citations" in event_names
    assert "token" in event_names
    assert "done" in event_names


@pytest.mark.asyncio
async def test_fallback_on_unanswerable_query(setup_pipeline):
    pipeline, _ = setup_pipeline

    # Completely unrelated query
    query = "What is the recipe for chocolate lava cake?"
    response = await pipeline.aquery(query)

    assert response.status in ("fallback", "insufficient_context")
    ans_lower = response.answer.lower()
    assert "sufficient information" in ans_lower or "cannot verify" in ans_lower


@pytest.mark.asyncio
async def test_benchmark_evaluator(setup_pipeline):
    pipeline, _ = setup_pipeline
    evaluator = BenchmarkEvaluator(pipeline)

    test_cases = [
        {
            "question": "What is the encryption algorithm for volumes at rest?",
            "ground_truth": "All storage volumes are encrypted at rest using AES-256-GCM.",
        },
        {
            "question": "How many availability zones are data replicated across?",
            "ground_truth": "Data is synchronously replicated across three Availability Zones.",
        },
    ]

    report = await evaluator.run_benchmark(test_cases)

    assert report["total_cases"] == 2
    assert report["avg_context_precision"] > 0.0
    assert report["avg_faithfulness"] >= 0.70
    assert report["avg_answer_relevance"] > 0.0
