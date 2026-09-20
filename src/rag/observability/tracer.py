"""Structured observability, telemetry, and distributed tracing for RAG pipelines."""

import asyncio
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterator
from uuid import uuid4
from pydantic import BaseModel, Field


class Span(BaseModel):
    """A single measured execution span in a RAG pipeline trace."""

    span_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: str = "ok"
    error_message: str | None = None


class Trace(BaseModel):
    """A collection of spans representing a single end-to-end request lifecycle."""

    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query: str = ""
    spans: list[Span] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineTracer:
    """Manages active trace spans with async context managers and memory buffers."""

    def __init__(self, max_history: int = 500) -> None:
        self.max_history = max_history
        self._traces: list[Trace] = []
        self._lock = asyncio.Lock()

    def start_trace(self, query: str, metadata: dict[str, Any] | None = None) -> Trace:
        """Initialize a new trace context for a query."""
        trace = Trace(
            query=query,
            metadata=metadata or {},
        )
        return trace

    @asynccontextmanager
    async def aspan(
        self,
        trace: Trace,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncIterator[Span]:
        """Record an asynchronous execution span within a trace."""
        t_start = time.perf_counter()
        span = Span(
            name=name,
            start_time=t_start,
            attributes=attributes or {},
        )
        try:
            yield span
        except Exception as e:
            span.status = "error"
            span.error_message = str(e)
            raise
        finally:
            t_end = time.perf_counter()
            span.end_time = t_end
            span.duration_ms = round((t_end - t_start) * 1000.0, 2)
            trace.spans.append(span)

    @contextmanager
    def span(
        self,
        trace: Trace,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """Record a synchronous execution span within a trace."""
        t_start = time.perf_counter()
        span = Span(
            name=name,
            start_time=t_start,
            attributes=attributes or {},
        )
        try:
            yield span
        except Exception as e:
            span.status = "error"
            span.error_message = str(e)
            raise
        finally:
            t_end = time.perf_counter()
            span.end_time = t_end
            span.duration_ms = round((t_end - t_start) * 1000.0, 2)
            trace.spans.append(span)

    async def record_trace(self, trace: Trace) -> None:
        """Store finished trace in buffer with FIFO eviction."""
        if trace.spans:
            trace.total_duration_ms = round(sum(s.duration_ms for s in trace.spans), 2)
        async with self._lock:
            if len(self._traces) >= self.max_history:
                self._traces.pop(0)
            self._traces.append(trace)

    async def get_recent_traces(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve recent traces for operational dashboards."""
        async with self._lock:
            return [t.model_dump() for t in reversed(self._traces[-limit:])]


# Global tracer singleton instance
tracer = PipelineTracer()

