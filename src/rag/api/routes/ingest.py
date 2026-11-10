"""Document ingestion and CDC management endpoints."""

from typing import Any
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from rag.config import settings
from pydantic import BaseModel, Field
from rag.ingestion.parsers.factory import ParserFactory
from rag.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


class IngestTextRequest(BaseModel):
    text: str = Field(description="Raw document content")
    title: str = Field(default="untitled_document", description="Document title")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata tags and ACLs")
    force_reindex: bool = False


@router.post("/text")
async def ingest_text_endpoint(req: IngestTextRequest, request: Request):
    """Ingest raw text or markdown content into the RAG vector store."""
    ingestion: IngestionPipeline = request.app.state.ingestion
    try:
        result = await ingestion.ingest_text(
            text=req.text,
            title=req.title,
            extra_metadata=req.metadata,
            force_reindex=req.force_reindex,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Document ingestion failed.") from e


@router.post("/file")
async def ingest_file_endpoint(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    acl_groups: str | None = Form(default=None),
    force_reindex: bool = Form(default=False),
):
    """Upload and ingest a document file (PDF, Markdown, Text)."""
    ingestion: IngestionPipeline = request.app.state.ingestion
    try:
        content_bytes = await file.read(settings.max_upload_bytes + 1)
        if len(content_bytes) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds the configured size limit.")
        filename = file.filename or "unknown_file"
        doc_title = title or filename

        extra_meta: dict[str, Any] = {}
        if acl_groups:
            extra_meta["acl_groups"] = [g.strip() for g in acl_groups.split(",") if g.strip()]

        parser = ParserFactory.get_parser(filename)
        document = parser.parse(
            content=content_bytes,
            filename=filename,
            extra_metadata=extra_meta,
        )
        if title:
            document.metadata.title = title

        result = await ingestion.ingest_document(document, force_reindex=force_reindex)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Document ingestion failed.") from e


@router.delete("/documents/{doc_id}")
async def delete_document_endpoint(doc_id: str, request: Request):
    """Delete all chunks for a document from the vector store (CDC purge)."""
    ingestion: IngestionPipeline = request.app.state.ingestion
    try:
        await ingestion.vector_store.delete_by_doc_id(doc_id)
        return {"doc_id": doc_id, "status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Document deletion failed.") from e

