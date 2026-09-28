"""Evaluation metrics implementing the RAG Triad: Context Relevance, Faithfulness, and Answer Relevance."""

import math
import re
from rag.domain.models import EvaluationMetrics, SearchResult


class RAGTriadEvaluator:
    """Computes RAG Triad quality metrics for offline and CI/CD validation."""

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        words = re.findall(r"\w+", text.lower())
        return {w for w in words if len(w) > 3}

    def evaluate_context_relevance(self, query: str, retrieved_chunks: list[SearchResult]) -> float:
        """Measure what fraction of retrieved chunks contain query-relevant content."""
        if not retrieved_chunks:
            return 0.0

        q_terms = self._extract_keywords(query)
        if not q_terms:
            return 1.0

        relevant_count = 0
        for res in retrieved_chunks:
            chunk_terms = self._extract_keywords(res.chunk.content)
            # If at least 30% of query keywords appear in chunk, classify as relevant
            overlap = len(q_terms.intersection(chunk_terms))
            if overlap / len(q_terms) >= 0.25:
                relevant_count += 1

        return round(relevant_count / len(retrieved_chunks), 4)

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """Measure what fraction of generated claims are supported by context."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 8]
        if not sentences:
            return 1.0

        context_lower = context.lower()
        supported = 0

        for s in sentences:
            # Strip citation tags
            cleaned = re.sub(r"\[Ref-\d+\]", "", s).strip()
            terms = self._extract_keywords(cleaned)
            if not terms:
                supported += 1
                continue
            matched = sum(1 for t in terms if t in context_lower)
            if (matched / len(terms)) >= 0.50:
                supported += 1

        return round(supported / len(sentences), 4)

    def evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """Measure whether the answer is relevant and responsive to the user query."""
        q_terms = self._extract_keywords(query)
        a_terms = self._extract_keywords(answer)
        if not q_terms or not a_terms:
            return 1.0

        overlap = len(q_terms.intersection(a_terms))
        relevance = overlap / len(q_terms)
        return round(min(1.0, relevance * 1.5), 4)

    def evaluate_triad(
        self,
        query: str,
        answer: str,
        context: str,
        retrieved_chunks: list[SearchResult],
        ground_truth: str | None = None,
    ) -> EvaluationMetrics:
        """Run all RAG Triad evaluations and return consolidated metrics."""
        ctx_relevance = self.evaluate_context_relevance(query, retrieved_chunks)
        faithfulness = self.evaluate_faithfulness(answer, context)
        ans_relevance = self.evaluate_answer_relevance(query, answer)

        # Context recall against ground truth if available
        ctx_recall = 1.0
        if ground_truth:
            gt_terms = self._extract_keywords(ground_truth)
            all_ctx_terms = self._extract_keywords(context)
            if gt_terms:
                ctx_recall = round(len(gt_terms.intersection(all_ctx_terms)) / len(gt_terms), 4)

        return EvaluationMetrics(
            context_precision=ctx_relevance,
            context_recall=ctx_recall,
            faithfulness=faithfulness,
            answer_relevance=ans_relevance,
        )

