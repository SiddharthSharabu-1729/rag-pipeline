"""Tests for vector storage, sparse encoding, and semantic cache."""

import pytest
from rag.domain.models import Chunk, Query, RAGResponse
from rag.storage.embeddings import MockEmbeddingClient
from rag.storage.qdrant_store import QdrantVectorStore
from rag.storage.semantic_cache import SemanticCache
from rag.storage.sparse import SparseBM25Encoder


def test_sparse_bm25_encoder():
    encoder = SparseBM25Encoder()
    text = "NVIDIA RTX-4090 GPU benchmark with CVE-2024-38077 vulnerability."
    sparse_vec = encoder.encode(text)

    assert len(sparse_vec.indices) > 0
    assert len(sparse_vec.indices) == len(sparse_vec.values)
    # Check that indices are positive integers
    assert all(isinstance(idx, int) and idx > 0 for idx in sparse_vec.indices)
    assert all(val > 0.0 for val in sparse_vec.values)


@pytest.mark.asyncio
async def test_mock_embedding_client_normalization():
    client = MockEmbeddingClient(dimension=128)
    vec = await client.aembed_query("production rag pipeline")

    assert len(vec) == 128
    # Test L2 norm ~ 1.0
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4


@pytest.mark.asyncio
async def test_qdrant_store_hybrid_crud():
    store = QdrantVectorStore(collection_name="test_qdrant_hybrid", dense_dim=64)
    await store.initialize()

    embedder = MockEmbeddingClient(dimension=64)
    v1 = await embedder.aembed_query("Database replication and failover")
    v2 = await embedder.aembed_query("Microservices authentication using OAuth2")

    chunks = [
        Chunk(
            chunk_id="chunk-1",
            doc_id="doc-db",
            content="Database replication and high availability failover procedures.",
            dense_vector=v1,
        ),
        Chunk(
            chunk_id="chunk-2",
            doc_id="doc-auth",
            content="Microservices authentication using OAuth2 and JWT tokens.",
            dense_vector=v2,
        ),
    ]

    # Test Upsert
    count = await store.upsert_chunks(chunks)
    assert count == 2

    # Test Dense Search
    query = Query(text="How does database failover work?")
    q_vec = await embedder.aembed_query(query.text)
    dense_res = await store.search_dense(q_vec, query, limit=5)
    assert len(dense_res) > 0
    assert dense_res[0].chunk.doc_id == "doc-db"

    # Test Sparse Search
    query_sparse = Query(text="OAuth2 JWT")
    sparse_res = await store.search_sparse(query_sparse, limit=5)
    assert len(sparse_res) > 0
    assert sparse_res[0].chunk.doc_id == "doc-auth"

    # Test Hybrid Search (RRF)
    hybrid_res = await store.search_hybrid(query_sparse, q_vec, limit=5)
    assert len(hybrid_res) > 0

    # Test Delete by Doc ID
    del_res = await store.delete_by_doc_id("doc-db")
    assert del_res == 1


@pytest.mark.asyncio
async def test_semantic_cache():
    cache = SemanticCache(threshold=0.90)
    embedder = MockEmbeddingClient(dimension=64)

    q1_text = "What is the holiday leave policy?"
    q1_vec = await embedder.aembed_query(q1_text)

    response = RAGResponse(
        query=q1_text,
        answer="Employees receive 20 days of paid annual leave.",
        from_cache=False,
    )

    await cache.put(q1_text, q1_vec, response)
    assert await cache.size() == 1

    # Exact query hit
    hit = await cache.get(q1_vec)
    assert hit is not None
    assert hit.from_cache is True
    assert hit.answer == response.answer

    # Different query miss
    diff_vec = await embedder.aembed_query("Quantum computing entanglement")
    miss = await cache.get(diff_vec)
    assert miss is None



@pytest.mark.asyncio
async def test_semantic_cache_isolated_by_context():
    cache = SemanticCache(threshold=0.90)
    embedder = MockEmbeddingClient(dimension=64)
    vector = await embedder.aembed_query("What is the holiday leave policy?")
    response = RAGResponse(query="leave policy", answer="Private answer")
    await cache.put("leave policy", vector, response, context_key="tenant-a")
    assert await cache.get(vector, context_key="tenant-b") is None
    hit = await cache.get(vector, context_key="tenant-a")
    assert hit is not None and hit.answer == "Private answer"


@pytest.mark.asyncio
async def test_semantic_cache_rejects_different_vector_dimensions():
    cache = SemanticCache(threshold=0.90)
    response = RAGResponse(query="question", answer="answer")
    await cache.put("question", [1.0, 0.0], response)
    assert await cache.get([1.0, 0.0, 0.0]) is None


def test_qdrant_acl_filter_for_ungrouped_caller_only_matches_public():
    store = QdrantVectorStore(collection_name="acl-filter-test", dense_dim=4)
    q_filter = store._build_filter(Query(text="public data"))
    assert q_filter is not None
    assert q_filter.should is not None and len(q_filter.should) == 1
