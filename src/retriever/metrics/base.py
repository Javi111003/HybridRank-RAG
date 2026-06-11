from abc import ABC, abstractmethod
from typing import Any, TypeAlias

from src.retriever.types import RetrievalResults

DocumentIds: TypeAlias = list[str]
MetricResult: TypeAlias = dict[str, Any]


class Metric(ABC):
    """Base contract for retrieval evaluation metrics."""

    @abstractmethod
    def compute(
        self,
        retrieved_documents: RetrievalResults,
        relevant_documents: DocumentIds,
        k: int | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


def effective_k(retrieved_documents: RetrievalResults, k: int | None) -> int:
    if k is None:
        return len(retrieved_documents)
    return max(0, min(k, len(retrieved_documents)))


def metric_name(base_name: str, requested_k: int | None, used_k: int) -> str:
    return f"{base_name}@{used_k}" if requested_k is not None else base_name


def top_k_ids(retrieved_documents: RetrievalResults, k: int) -> set[str]:
    return {doc_id for doc_id, _ in retrieved_documents[:k]}
