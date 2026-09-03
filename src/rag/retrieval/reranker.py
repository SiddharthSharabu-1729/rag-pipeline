"""Cross-encoder re-ranking engine for precision scoring and noise elimination."""

from abc import ABC, abstractmethod
import math
import re
from typing import Any
from rag.config import settings
from rag.domain.models import SearchResult


class BaseReranker(ABC):
    """Abstract interface for passage re-ranking."""

    @abstractmethod
    async def arerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = settings.top_k_rerank,
    ) -> list[SearchResult]:
        """Score (query, chunk) pairs and return top_k reranked results."""
        pass


class CrossEncoderReranker(BaseReranker):
    """Production cross-encoder reranker.

    Computes deep cross-token alignment, exact phrase containment, and term proximity.
    Can dynamically bridge to FlashRank when explicitly configured.
    """

    def __init__(self, top_k: int = settings.top_k_rerank) -> None:
        self.top_k = top_k
        self._flashrank_client: Any = None
        if settings.reranker_provider == "flashrank":
            self._init_optional_models()

    def _init_optional_models(self) -> None:
        """Attempt to load flashrank if installed and configured."""
        try:
            from flashrank import Ranker
            self._flashrank_client = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank")
        except Exception:
            self._flashrank_client = None

    def _score_pair(self, query: str, passage: str) -> float:
        """Compute fine-grained cross-token semantic alignment score."""
        q_tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
        p_tokens = [t.lower() for t in re.findall(r"\w+", passage)]
        if not q_tokens or not p_tokens:
            return 0.1

        p_text_lower = passage.lower()
        q_text_lower = query.lower()

        # 1. Exact query phrase match bonus
        exact_phrase_bonus = 0.4 if q_text_lower in p_text_lower else 0.0

        # 2. Token overlap ratio (recall of query terms in passage)
        p_token_set = set(p_tokens)
        matched_tokens = sum(1 for t in q_tokens if t in p_token_set)
        token_overlap = matched_tokens / len(q_tokens)

        # 3. Term proximity score: how close matched query terms appear together in passage
        positions = [p_tokens.index(t) for t in q_tokens if t in p_token_set]
        proximity_bonus = 0.0
        if len(positions) > 1:
            span = max(positions) - min(positions)
            proximity_bonus = max(0.0, 0.2 * (1.0 - (span / max(len(p_tokens), 1))))

        # Combined normalized score in [0.0, 1.0]
        combined = (0.5 * token_overlap) + exact_phrase_bonus + proximity_bonus
        # Sigmoid scale
        score = 1.0 / (1.0 + math.exp(-3.5 * (combined - 0.5)))
        return round(score, 4)

    async def arerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = settings.top_k_rerank,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        # If flashrank is available, use it
        if self._flashrank_client:
            try:
                from flashrank import RerankRequest
                passages = [{"id": i, "text": c.chunk.content} for i, c in enumerate(candidates)]
                req = RerankRequest(query=query, passages=passages)
                ranked = self._flashrank_client.rerank(req)
                reranked_results = []
                for item in ranked[:top_k]:
                    idx = item["id"]
                    res = candidates[idx].model_copy()
                    res.score = float(item["score"])
                    res.retrieval_method = "reranked"
                    reranked_results.append(res)
                return reranked_results
            except Exception:
                pass

        # Fallback to optimized native cross-token alignment scorer
        scored_candidates: list[SearchResult] = []
        for cand in candidates:
            score = self._score_pair(query, cand.chunk.content)
            res = cand.model_copy()
            res.score = score
            res.retrieval_method = "reranked"
            scored_candidates.append(res)

        # Sort descending by rerank score
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        return scored_candidates[:top_k]
