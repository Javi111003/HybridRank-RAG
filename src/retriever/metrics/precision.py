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


class PrecisionAtK(Metric):
    """
    Purity of the top-k results.

    Precision@k = relevant_found_in_top_k / k. In RAG, high precision keeps
    irrelevant fragments out of the prompt context and reduces generation noise.
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
        score = found / used_k if used_k else 0.0

        return {
            "score": score,
            "metric_name": metric_name("Precision", k, used_k),
            "k": used_k,
            "relevant_found": found,
            "total_retrieved": used_k,
        }

    @property
    def name(self) -> str:
        return "Precision@k"
