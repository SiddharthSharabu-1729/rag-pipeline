"""Context window optimizer: Lost-in-the-middle mitigation, parent expansion, and quality gating."""

from rag.config import settings
from rag.domain.models import Citation, SearchResult


class ContextOptimizer:
    """Optimizes retrieved chunks for LLM attention, relevance thresholds, and token efficiency."""

    def __init__(self, relevance_threshold: float = settings.relevance_threshold) -> None:
        self.relevance_threshold = relevance_threshold

    def evaluate_quality_gate(self, candidates: list[SearchResult]) -> bool:
        """Return True if at least one candidate exceeds the confidence relevance threshold."""
        if not candidates:
            return False
        max_score = max(c.score for c in candidates)
        return max_score >= self.relevance_threshold

    def mitigate_lost_in_the_middle(self, candidates: list[SearchResult]) -> list[SearchResult]:
        """Reorder ranked candidates so highest-scoring chunks reside at window boundaries.

        Order becomes: [Rank 1, Rank 3, ..., Rank 4, Rank 2]
        """
        if len(candidates) <= 2:
            return candidates

        reordered: list[SearchResult] = [None] * len(candidates)  # type: ignore[list-item]
        left = 0
        right = len(candidates) - 1

        for i, cand in enumerate(candidates):
            if i % 2 == 0:
                reordered[left] = cand
                left += 1
            else:
                reordered[right] = cand
                right -= 1

        return reordered

    def assemble_context_and_citations(
        self,
        candidates: list[SearchResult],
    ) -> tuple[str, list[Citation]]:
        """Deduplicate parent contexts and format reference context and structured citations."""
        if not candidates:
            return "", []

        seen_parents: set[str] = set()
        formatted_blocks: list[str] = []
        citations: list[Citation] = []

        for idx, res in enumerate(candidates, start=1):
            chunk = res.chunk
            parent_key = chunk.parent_id or chunk.chunk_id

            # Content to present to LLM: parent context if available, otherwise leaf content
            display_text = chunk.parent_content or chunk.content

            # Create citation reference
            title = chunk.metadata.get("title", f"Document-{chunk.doc_id[:8]}")
            source_uri = chunk.metadata.get("source_uri", "unknown")
            snippet = chunk.content[:200] + ("..." if len(chunk.content) > 200 else "")

            citation = Citation(
                source_id=f"Ref-{idx}",
                title=title,
                source_uri=source_uri,
                page_number=chunk.page_number,
                snippet=snippet,
            )
            citations.append(citation)

            # Avoid repeating identical parent text block in context
            if parent_key in seen_parents:
                continue
            seen_parents.add(parent_key)

            page_info = f" | Page {chunk.page_number}" if chunk.page_number else ""
            block = (
                f"--- [Ref-{idx} | {title}{page_info}] ---\n"
                f"{display_text.strip()}\n"
            )
            formatted_blocks.append(block)

        full_context_str = "\n".join(formatted_blocks)
        return full_context_str, citations

