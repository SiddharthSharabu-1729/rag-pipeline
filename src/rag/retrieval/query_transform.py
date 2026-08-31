"""Pre-retrieval query understanding, sanitization, and transformations."""

import re
from typing import Any
from rag.domain.models import Query


class QueryTransformer:
    """Sanitizes user queries and handles conversational contextualization and sub-queries."""

    @staticmethod
    def sanitize(text: str) -> str:
        """Strip dangerous prompt injection payloads and normalize whitespace."""
        # Remove prompt injection delimiters and zero-width spaces
        cleaned = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
        cleaned = re.sub(r"(?i)(ignore previous instructions|disregard above|system prompt)", "", cleaned)
        cleaned = " ".join(cleaned.split())
        return cleaned.strip()

    @staticmethod
    def contextualize_query(query_text: str, history: list[dict[str, str]] | None) -> str:
        """Resolve pronouns and co-references using conversational dialogue history."""
        if not history:
            return query_text

        # If question contains pronouns referencing prior turns ("it", "they", "that", "the former")
        pronoun_pattern = re.compile(r"\b(it|its|they|them|their|that|this|these|those|the former|the latter)\b", re.I)
        if pronoun_pattern.search(query_text):
            # Extract last user query or key entities from previous exchange
            last_turns = [t["content"] for t in history[-2:] if t.get("role") in ("user", "assistant")]
            if last_turns:
                context_prefix = " ".join(last_turns[-1].split()[:12])
                return f"{query_text} (Context: regarding {context_prefix})"

        return query_text

    @staticmethod
    def decompose_query(query_text: str) -> list[str]:
        """Decompose multi-hop or comparative queries into atomic sub-queries."""
        # Detect comparisons or multiple clauses joined by 'and', 'vs', 'versus', 'compared to'
        compare_match = re.search(r"(.+?)\s+(?:versus|vs\.?|compared to)\s+(.+)", query_text, flags=re.I)
        if compare_match:
            entity_a = compare_match.group(1).strip()
            entity_b = compare_match.group(2).strip()
            return [
                query_text,
                f"Features and details of {entity_a}",
                f"Features and details of {entity_b}",
            ]

        # Multi-question split
        if "?" in query_text and len(query_text.split("?")) > 2:
            parts = [p.strip() + "?" for p in query_text.split("?") if p.strip()]
            return [query_text] + parts

        return [query_text]

    @staticmethod
    def extract_filters(query_text: str) -> dict[str, Any]:
        """Extract explicit metadata filters such as year, format, or category from query text."""
        filters: dict[str, Any] = {}

        # Extract 4-digit year if referenced (e.g., 2023, 2024, 2025, 2026)
        year_match = re.search(r"\b(20[1-3][0-9])\b", query_text)
        if year_match:
            filters["year"] = int(year_match.group(1))

        # Extract format hints
        if "pdf" in query_text.lower():
            filters["format"] = "pdf"
        elif "markdown" in query_text.lower():
            filters["format"] = "markdown"

        return filters

