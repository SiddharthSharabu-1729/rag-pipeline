"""Interactive Command Line Interface (CLI) for the Production RAG Pipeline."""

import argparse
import asyncio
import sys
from pathlib import Path
from rag.config import settings
from rag.domain.models import Document, DocumentMetadata
from rag.evaluation.evaluator import BenchmarkEvaluator
from rag.ingestion.pipeline import IngestionPipeline
from rag.observability.tracer import tracer
from rag.pipeline.rag_pipeline import RAGPipeline
from rag.storage.embeddings import EmbeddingFactory
from rag.storage.qdrant_store import QdrantVectorStore


async def _run_query(query_text: str) -> None:
    """Execute a query through the RAG pipeline and display formatted output."""
    embedder = EmbeddingFactory.get_client()
    store = QdrantVectorStore(dense_dim=embedder.dimension)
    await store.initialize()
    pipeline = RAGPipeline(vector_store=store, embedding_client=embedder)

    print(f"\n🔍 Query: {query_text}")
    print("⏳ Executing hybrid retrieval, reranking, and grounded generation...\n")

    resp = await pipeline.aquery(query_text)

    print("=" * 60)
    print("📝 GROUNDED ANSWER:")
    print("=" * 60)
    print(resp.answer)
    print("\n" + "=" * 60)
    print(f"📚 VERIFIED CITATIONS ({len(resp.citations)}):")
    print("=" * 60)
    for c in resp.citations:
        page_str = f" | Page {c.page_number}" if c.page_number else ""
        print(f"[{c.source_id}] {c.title}{page_str}")
        print(f"    URI: {c.source_uri}")
        print(f"    Excerpt: {c.snippet}\n")

    print("=" * 60)
    print("⚡ LATENCY BREAKDOWN:")
    print("=" * 60)
    lat = resp.latency
    print(f"  Cache Check:     {lat.cache_check_ms:.2f} ms (Hit: {resp.from_cache})")
    print(f"  Hybrid Retrieve: {lat.dense_retrieval_ms:.2f} ms")
    print(f"  Cross Rerank:    {lat.fusion_rerank_ms:.2f} ms")
    print(f"  LLM Generation:  {lat.generation_ms:.2f} ms")
    print(f"  Self-RAG Verify: {lat.self_rag_verification_ms:.2f} ms")
    print(f"  TOTAL LATENCY:   {lat.total_latency_ms:.2f} ms")
    if resp.trace_id:
        print(f"  TRACE ID:        {resp.trace_id}")
    print("=" * 60 + "\n")


async def _run_ingest(file_path: str, title: str | None) -> None:
    """Ingest a local document file into the vector store."""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    embedder = EmbeddingFactory.get_client()
    store = QdrantVectorStore(dense_dim=embedder.dimension)
    await store.initialize()
    ingestion = IngestionPipeline(vector_store=store, embedding_client=embedder)

    print(f"📄 Ingesting {path.name}...")
    res = await ingestion.ingest_file(path, extra_metadata={"title": title or path.name})
    print(f"✅ Status: {res['status']}")
    print(f"📦 Chunks Indexed: {res['chunks_count']}")
    print(f"🔑 Document ID: {res['doc_id']}")


async def _run_benchmark() -> None:
    """Run golden test suite benchmark."""
    embedder = EmbeddingFactory.get_client()
    store = QdrantVectorStore(dense_dim=embedder.dimension)
    await store.initialize()
    pipeline = RAGPipeline(vector_store=store, embedding_client=embedder)
    evaluator = BenchmarkEvaluator(pipeline)

    # Ingest baseline test docs
    doc = Document(
        id="bench-doc-01",
        content="Kubernetes pod autoscaling scales replicas based on CPU utilization.\nTarget threshold is 80% CPU.",
        metadata=DocumentMetadata(source_uri="bench.md", title="K8s Autoscaling"),
    )
    ingestion = IngestionPipeline(vector_store=store, embedding_client=embedder)
    await ingestion.ingest_document(doc)

    test_cases = [
        {"question": "How does pod autoscaling work?", "ground_truth": "scales replicas based on CPU utilization"},
        {"question": "What is the CPU target threshold?", "ground_truth": "80% CPU"},
    ]

    print("\n🧪 Running automated RAG Triad evaluation...")
    summary = await evaluator.run_benchmark(test_cases)
    print("\n" + "=" * 60)
    print("📊 RAG TRIAD BENCHMARK REPORT:")
    print("=" * 60)
    print(f"  Total Cases:         {summary['total_cases']}")
    print(f"  Context Precision:   {summary['avg_context_precision'] * 100:.1f}%")
    print(f"  Context Recall:      {summary['avg_context_recall'] * 100:.1f}%")
    print(f"  Faithfulness:        {summary['avg_faithfulness'] * 100:.1f}%")
    print(f"  Answer Relevance:    {summary['avg_answer_relevance'] * 100:.1f}%")
    print(f"  Average Latency:     {summary['avg_latency_ms']:.2f} ms")
    print("=" * 60 + "\n")


def main() -> None:
    """CLI dispatcher."""
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Production-Grade Enterprise RAG Pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI REST and SSE server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")

    # query command
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline from terminal")
    query_parser.add_argument("query_text", help="Question text")

    # ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document into the vector store")
    ingest_parser.add_argument("file_path", help="Path to Markdown, PDF, or Text file")
    ingest_parser.add_argument("--title", default=None, help="Document title")

    # benchmark command
    subparsers.add_parser("benchmark", help="Run RAG Triad benchmark suite")

    args = parser.parse_args()

    if args.command == "serve" or args.command is None:
        import uvicorn
        host = getattr(args, "host", "0.0.0.0")
        port = getattr(args, "port", 8000)
        print(f"\n🚀 Starting {settings.app_name} v{settings.app_version}")
        print(f"🌐 OpenAPI Documentation: http://{host}:{port}/docs")
        print(f"❤️ Health Check: http://{host}:{port}/api/v1/health\n")
        uvicorn.run("rag.api.app:app", host=host, port=port, reload=False)

    elif args.command == "query":
        asyncio.run(_run_query(args.query_text))

    elif args.command == "ingest":
        asyncio.run(_run_ingest(args.file_path, args.title))

    elif args.command == "benchmark":
        asyncio.run(_run_benchmark())


if __name__ == "__main__":
    main()

