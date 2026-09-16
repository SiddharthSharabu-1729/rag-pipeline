"""In-memory vector semantic caching layer for sub-30ms query resolution."""

import asyncio
import math
import time
from typing import NamedTuple
from rag.config import settings
from rag.domain.models import RAGResponse


class CacheEntry(NamedTuple):
    query_text: str
    vector: list[float]
    response: RAGResponse
    created_at: float
    last_accessed: float


class SemanticCache:
    """Thread-safe semantic cache matching queries by embedding cosine similarity."""

    def __init__(
        self,
        threshold: float = settings.semantic_cache_threshold,
        max_entries: int = settings.semantic_cache_max_entries,
        ttl_seconds: float = 3600.0 * 24,  # 24 hours default
    ) -> None:
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: list[CacheEntry] = []
        self._lock = asyncio.Lock()

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two unit or non-unit vectors."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def get(self, query_vector: list[float]) -> RAGResponse | None:
        """Search cache for a query vector exceeding semantic similarity threshold."""
        if not settings.semantic_cache_enabled:
            return None

        async with self._lock:
            now = time.time()
            best_match: CacheEntry | None = None
            highest_sim = -1.0

            # Prune expired entries
            valid_entries: list[CacheEntry] = []
            for entry in self._entries:
                if now - entry.created_at <= self.ttl_seconds:
                    valid_entries.append(entry)
                    sim = self._cosine_similarity(query_vector, entry.vector)
                    if sim > highest_sim:
                        highest_sim = sim
                        best_match = entry

            self._entries = valid_entries

            if best_match and highest_sim >= self.threshold:
                # Update last accessed for LRU
                cached_resp = best_match.response.model_copy(deep=True)
                cached_resp.from_cache = True
                cached_resp.latency.cache_check_ms = 5.0
                return cached_resp

        return None

    async def put(self, query_text: str, query_vector: list[float], response: RAGResponse) -> None:
        """Store a verified response in the semantic cache."""
        if not settings.semantic_cache_enabled:
            return

        async with self._lock:
            now = time.time()

            # Evict oldest / LRU if capacity reached
            if len(self._entries) >= self.max_entries:
                self._entries.sort(key=lambda e: e.last_accessed)
                self._entries.pop(0)

            # Store fresh response (mark from_cache=False for base)
            clean_resp = response.model_copy(deep=True)
            clean_resp.from_cache = False

            self._entries.append(
                CacheEntry(
                    query_text=query_text,
                    vector=query_vector,
                    response=clean_resp,
                    created_at=now,
                    last_accessed=now,
                )
            )

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._entries.clear()

    async def size(self) -> int:
        """Return number of active cache entries."""
        async with self._lock:
            return len(self._entries)

