from typing import Any

from src.retriever.types import RetrievalResults

from .base import DocumentIds, Metric, MetricResult, effective_k


class MRR(Metric):
    """
    Position of the first relevant result.

    MRR = 1 / rank(first_relevant), with ranks starting at 1. It is useful
    when a single early relevant fragment is enough for the task.
    """

    def compute(
        self,
        retrieved_documents: RetrievalResults,
        relevant_documents: DocumentIds,
        k: int | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        relevant = set(relevant_documents)
        if not retrieved_documents or not relevant:
            return self._result(first_relevant_rank=None)

        for rank, (doc_id, _) in enumerate(
            retrieved_documents[: effective_k(retrieved_documents, k)],
            start=1,
        ):
            if doc_id in relevant:
                return self._result(first_relevant_rank=rank)

        return self._result(first_relevant_rank=None)

    @staticmethod
    def _result(first_relevant_rank: int | None) -> MetricResult:
        return {
            "score": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
            "metric_name": "MRR",
            "first_relevant_rank": first_relevant_rank,
        }

    @property
    def name(self) -> str:
        return "MRR"
