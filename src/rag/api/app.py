"""FastAPI application for the Production RAG Pipeline."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag.api.routes import health, ingest, query
from rag.config import settings
from rag.ingestion.pipeline import IngestionPipeline
from rag.pipeline.rag_pipeline import RAGPipeline
from rag.storage.embeddings import EmbeddingFactory
from rag.storage.qdrant_store import QdrantVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize storage, embeddings, and pipeline components."""
    embedding_client = EmbeddingFactory.get_client()
    vector_store = QdrantVectorStore(dense_dim=embedding_client.dimension)
    await vector_store.initialize()

    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_client=embedding_client,
    )
    ingestion = IngestionPipeline(
        vector_store=vector_store,
        embedding_client=embedding_client,
    )

    app.state.vector_store = vector_store
    app.state.pipeline = pipeline
    app.state.ingestion = ingestion

    yield

    # Clean up resources if necessary


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise-grade production RAG pipeline with hybrid search, cross-encoder reranking, and Self-RAG verification.",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")


@app.get("/")
async def root():
    """Service metadata and API documentation discovery endpoint."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs_url": "/docs",
        "health_url": "/api/v1/health",
        "openapi_url": "/openapi.json",
    }

