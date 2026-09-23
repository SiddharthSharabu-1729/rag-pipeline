"""Health check and diagnostics route."""

from fastapi import APIRouter, Depends
from rag.config import settings
from rag.observability.tracer import tracer

router = APIRouter(tags=["Health & Telemetry"])


@router.get("/health")
async def health_check():
    """Health status and configuration overview."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "embedding_provider": settings.embedding_provider,
        "llm_provider": settings.llm_provider,
        "semantic_cache_enabled": settings.semantic_cache_enabled,
    }


@router.get("/traces")
async def get_traces(limit: int = 20):
    """Retrieve operational distributed trace spans."""
    recent = await tracer.get_recent_traces(limit=limit)
    return {"traces": recent, "count": len(recent)}

