from abc import ABC, abstractmethod

class Reranker(ABC):

    @abstractmethod
    def rerank(self, query: str, candidates):
        pass