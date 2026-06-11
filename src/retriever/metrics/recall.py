from typing import Any

from src.retriever.types import RetrievalResults

from .base import (
    DocumentIds,
    Metric,
    MetricResult,
    effective_k,
    metric_name,
    top_k_ids,
)


class RecallAtK(Metric):
    """
    Coverage of relevant documents in the top-k results.

    Recall@k = relevant_found_in_top_k / total_relevant.
    In RAG, high recall means the generator is more likely to receive useful
    evidence, even if the context also contains extra noise.
    """

    def compute(
        self,
        retrieved_documents: RetrievalResults,
        relevant_documents: DocumentIds,
        k: int | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        used_k = effective_k(retrieved_documents, k)
        relevant = set(relevant_documents)
        found = len(top_k_ids(retrieved_documents, used_k) & relevant)
        score = found / len(relevant) if relevant else 0.0

        return {
            "score": score,
            "metric_name": metric_name("Recall", k, used_k),
            "k": used_k,
            "relevant_found": found,
            "total_relevant": len(relevant_documents),
        }

    @property
    def name(self) -> str:
        return "Recall@k"
