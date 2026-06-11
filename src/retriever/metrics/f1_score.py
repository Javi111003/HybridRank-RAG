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


class F1AtK(Metric):
    """
    Harmonic mean between Precision@k and Recall@k.

    F1@k = 2 * precision * recall / (precision + recall). It rewards rankings
    that balance useful coverage with a clean top-k context.
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

        precision = found / used_k if used_k else 0.0
        recall = found / len(relevant) if relevant else 0.0
        score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        return {
            "score": score,
            "metric_name": metric_name("F1", k, used_k),
            "k": used_k,
            "precision": precision,
            "recall": recall,
            "relevant_found": found,
            "total_relevant": len(relevant_documents),
            "total_retrieved": used_k,
        }

    @property
    def name(self) -> str:
        return "F1@k"
