from typing import Any

from src.retriever.types import RetrievalResults

from .base import DocumentIds, Metric, MetricResult, effective_k, metric_name


class MAP(Metric):
    """
    Average precision across relevant positions.

    AP = (1 / total_relevant) * sum(P@i for each relevant result at rank i).
    MAP is the mean AP over many queries; this class returns AP for one query.
    """

    def compute(
        self,
        retrieved_documents: RetrievalResults,
        relevant_documents: DocumentIds,
        k: int | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        relevant = set(relevant_documents)
        used_k = effective_k(retrieved_documents, k)

        if not retrieved_documents or not relevant:
            return self._result(metric_name("MAP", k, used_k), [], 0)

        precisions_at_relevant: list[float] = []
        relevant_seen = 0

        for rank, (doc_id, _) in enumerate(retrieved_documents[:used_k], start=1):
            if doc_id in relevant:
                relevant_seen += 1
                precisions_at_relevant.append(relevant_seen / rank)

        return self._result(
            metric_name("MAP", k, used_k),
            precisions_at_relevant,
            relevant_seen,
            total_relevant=len(relevant_documents),
        )

    @staticmethod
    def _result(
        name: str,
        precisions_at_relevant: list[float],
        relevant_seen: int,
        total_relevant: int = 0,
    ) -> MetricResult:
        average_precision = (
            sum(precisions_at_relevant) / total_relevant
            if total_relevant
            else 0.0
        )
        return {
            "score": average_precision,
            "metric_name": name,
            "average_precision": average_precision,
            "precisions_at_relevant": precisions_at_relevant,
            "num_relevant_retrieved": relevant_seen,
        }

    @property
    def name(self) -> str:
        return "MAP"
