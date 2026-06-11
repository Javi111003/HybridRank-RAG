from collections.abc import Mapping

from .retriever import Retriever
from .fusion.base import FusionStrategy
from .types import ResultsByRetriever, RetrievalResults


class HybridRetriever(Retriever):
    """Runs several retrievers and fuses their rankings."""

    def __init__(
        self,
        retrievers: Mapping[str, Retriever],
        fusion_strategy: FusionStrategy,
        candidate_k: int = 50,
    ):
        if not retrievers:
            raise ValueError("Debe proporcionar al menos un recuperador")
        if candidate_k <= 0:
            raise ValueError(f"candidate_k debe ser > 0, recibido: {candidate_k}")

        self._retrievers = dict(retrievers)
        self._fusion_strategy = fusion_strategy
        self._candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResults:
        if top_k <= 0:
            return []

        results_by_retriever: ResultsByRetriever = {}
        for name, retriever in self._retrievers.items():
            results_by_retriever[name] = retriever.retrieve(query, self._candidate_k)

        return self._fusion_strategy.fuse(results_by_retriever, top_k)

    @property
    def name(self) -> str:
        strategy_name = self._fusion_strategy.__class__.__name__
        retriever_names = "+".join(self._retrievers.keys())
        return f"HybridRetriever({retriever_names}|{strategy_name})"
