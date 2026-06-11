from math import log2
from typing import Any

from src.retriever.types import RetrievalResults

from .base import DocumentIds, Metric, MetricResult, effective_k


class NDCG(Metric):
    """
    Position-discounted ranking quality.

    DCG@k = sum((2^rel_i - 1) / log2(i + 1)); nDCG@k = DCG@k / IDCG@k.
    Uses binary relevance: 1 for relevant, 0 otherwise.
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

        if not retrieved_documents or not relevant:
            return self._result(used_k, dcg=0.0, idcg=0.0)

        gains = [
            1 if doc_id in relevant else 0
            for doc_id, _ in retrieved_documents[:used_k]
        ]
        dcg = _dcg(gains)

        ideal_relevant_count = min(len(relevant_documents), used_k)
        idcg = _dcg([1] * ideal_relevant_count)

        return self._result(used_k, dcg=dcg, idcg=idcg)

    @staticmethod
    def _result(k: int, dcg: float, idcg: float) -> MetricResult:
        return {
            "score": dcg / idcg if idcg else 0.0,
            "metric_name": f"nDCG@{k}",
            "k": k,
            "dcg": dcg,
            "idcg": idcg,
        }

    @property
    def name(self) -> str:
        return "nDCG@k"


def _dcg(relevances: list[int]) -> float:
    return sum(
        (2**relevance - 1) / log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )
