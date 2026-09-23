"""Query and streaming endpoints."""

import json
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from rag.domain.models import RAGResponse
from rag.pipeline.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/query", tags=["Query"])


class QueryRequest(BaseModel):
    query: str = Field(description="User question")
    session_id: str | None = None
    user_id: str | None = None
    acl_groups: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, str]] | None = None


@router.post("", response_model=RAGResponse)
async def query_endpoint(req: QueryRequest, request: Request) -> RAGResponse:
    """Execute synchronous-style query against the RAG pipeline."""
    pipeline: RAGPipeline = request.app.state.pipeline
    try:
        response = await pipeline.aquery(
            query_text=req.query,
            session_id=req.session_id,
            user_id=req.user_id,
            acl_groups=req.acl_groups,
            filters=req.filters,
            history=req.history,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/stream")
async def stream_query_endpoint(req: QueryRequest, request: Request):
    """Stream response tokens and citations via Server-Sent Events (SSE)."""
    pipeline: RAGPipeline = request.app.state.pipeline

    async def event_generator():
        try:
            async for event in pipeline.astream(
                query_text=req.query,
                acl_groups=req.acl_groups,
                filters=req.filters,
                history=req.history,
            ):
                event_name = event.get("event", "message")
                data_json = json.dumps(event.get("data", {}))
                yield f"event: {event_name}\ndata: {data_json}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

