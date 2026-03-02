from abc import ABC, abstractmethod
from typing import List, Tuple

class Retriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Returns a ranked list of (document_id, score)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass