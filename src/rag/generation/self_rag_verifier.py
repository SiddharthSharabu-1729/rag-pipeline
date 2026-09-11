"""Self-RAG / Corrective RAG (CRAG) verification and hallucination detection."""

import re
from rag.config import settings
from rag.domain.models import Citation, SearchResult


class SelfRAGVerifier:
    """Verifies generated responses against retrieved context for hallucination and citation fidelity."""

    @staticmethod
    def _extract_statements(text: str) -> list[str]:
        """Split generated text into atomic verifiable statements."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def verify_faithfulness(self, answer: str, context: str) -> tuple[float, list[str]]:
        """Evaluate what fraction of generated statements are grounded in context.

        Returns:
            (faithfulness_score, ungrounded_claims)
        """
        if not settings.enable_self_rag_verification:
            return 1.0, []

        statements = self._extract_statements(answer)
        if not statements:
            return 1.0, []

        context_lower = context.lower()
        ungrounded: list[str] = []
        grounded_count = 0

        for stmt in statements:
            # Clean statement from citations like [Ref-1]
            cleaned_stmt = re.sub(r"\[Ref-\d+\]", "", stmt).strip()
            # Extract significant keywords
            words = [w.lower() for w in re.findall(r"\w+", cleaned_stmt) if len(w) > 3]

            if not words:
                grounded_count += 1
                continue

            # Check overlap against context
            matched = sum(1 for w in words if w in context_lower)
            overlap_ratio = matched / len(words)

            # Statement is considered grounded if >= 60% of significant terms appear in context
            if overlap_ratio >= 0.60:
                grounded_count += 1
            else:
                ungrounded.append(stmt)

        score = grounded_count / len(statements)
        return round(score, 4), ungrounded

    def verify_citations(self, answer: str, citations: list[Citation]) -> bool:
        """Check whether every [Ref-X] cited in the answer actually exists in the citation list."""
        cited_ids = set(re.findall(r"\[(Ref-\d+)\]", answer))
        valid_ids = {c.source_id for c in citations}
        # If any cited ID is not in valid citations, citation hallucination occurred
        return cited_ids.issubset(valid_ids)

