"""Automated benchmark evaluator for regression testing and CI/CD pipelines."""

from typing import Any
from rag.domain.models import EvaluationMetrics
from rag.evaluation.metrics import RAGTriadEvaluator
from rag.pipeline.rag_pipeline import RAGPipeline


class BenchmarkEvaluator:
    """Runs automated RAG evaluation against a golden test suite."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline
        self.triad_evaluator = RAGTriadEvaluator()

    async def run_benchmark(
        self,
        test_cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute test cases and aggregate RAG Triad scores."""
        results: list[dict[str, Any]] = []

        total_precision = 0.0
        total_recall = 0.0
        total_faithfulness = 0.0
        total_relevance = 0.0
        total_latency = 0.0

        for case in test_cases:
            q_text = case["question"]
            ground_truth = case.get("ground_truth")

            resp = await self.pipeline.aquery(q_text)

            # Reconstruct context text from retrieved chunks
            ctx_text = "\n".join(c.chunk.content for c in resp.retrieved_chunks)

            metrics = self.triad_evaluator.evaluate_triad(
                query=q_text,
                answer=resp.answer,
                context=ctx_text,
                retrieved_chunks=resp.retrieved_chunks,
                ground_truth=ground_truth,
            )

            total_precision += metrics.context_precision
            total_recall += metrics.context_recall
            total_faithfulness += metrics.faithfulness
            total_relevance += metrics.answer_relevance
            total_latency += resp.latency.total_latency_ms

            results.append({
                "question": q_text,
                "answer": resp.answer,
                "metrics": metrics.model_dump(),
                "latency_ms": resp.latency.total_latency_ms,
                "from_cache": resp.from_cache,
            })

        n = max(len(test_cases), 1)
        summary = {
            "total_cases": len(test_cases),
            "avg_context_precision": round(total_precision / n, 4),
            "avg_context_recall": round(total_recall / n, 4),
            "avg_faithfulness": round(total_faithfulness / n, 4),
            "avg_answer_relevance": round(total_relevance / n, 4),
            "avg_latency_ms": round(total_latency / n, 2),
            "details": results,
        }
        return summary

