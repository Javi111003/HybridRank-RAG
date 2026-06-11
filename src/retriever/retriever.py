from abc import ABC, abstractmethod

from .types import RetrievalResults


class Retriever(ABC):
    """Base contract for ranked document retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> RetrievalResults:
        """Return ranked ``(document_id, score)`` pairs."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
