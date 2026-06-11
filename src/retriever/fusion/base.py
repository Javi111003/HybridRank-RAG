from abc import ABC, abstractmethod

from src.retriever.types import ResultsByRetriever, RetrievalResults


class FusionStrategy(ABC):
    """Combines ranked outputs from one or more retrievers."""

    @abstractmethod
    def fuse(
        self,
        results_by_retriever: ResultsByRetriever,
        top_k: int,
    ) -> RetrievalResults:
        raise NotImplementedError
