from .base import FusionStrategy
from .normalization import ScoreNormalizer, get_normalizer
from .utils import (
    get_all_doc_ids,
    require_results,
    results_to_rank_dict,
    results_to_score_dict,
    sort_and_truncate,
)
from src.retriever.types import ResultsByRetriever, RetrievalResults


def _normalize(
    results: RetrievalResults,
    normalizer: ScoreNormalizer,
) -> dict[str, float]:
    return normalizer.normalize(results_to_score_dict(results))


class HybridRankFusion(FusionStrategy):
    """HybridRank fusion combining weighted scores with RRF."""

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.2,
        k: int = 10,
        sparse_key: str = "bm25",
        dense_key: str = "dense",
        normalizer: str = "minmax",
        rrf_normalizer: str = "minmax",
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha debe estar en [0, 1], recibido: {alpha}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta debe estar en [0, 1], recibido: {beta}")
        if k <= 0:
            raise ValueError(f"k debe ser > 0, recibido: {k}")

        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.sparse_key = sparse_key
        self.dense_key = dense_key
        self.normalizer = get_normalizer(normalizer)
        self.rrf_normalizer = get_normalizer(rrf_normalizer)

    def fuse(
        self,
        results_by_retriever: ResultsByRetriever,
        top_k: int,
    ) -> RetrievalResults:
        if not results_by_retriever:
            return []

        sparse_results = require_results(
            results_by_retriever, self.sparse_key, "sparse"
        )
        dense_results = require_results(results_by_retriever, self.dense_key, "dense")

        all_docs = get_all_doc_ids(results_by_retriever)
        sparse_scores = _normalize(sparse_results, self.normalizer)
        dense_scores = _normalize(dense_results, self.normalizer)

        weighted_scores = {
            doc_id: (
                self.alpha * sparse_scores.get(doc_id, 0.0)
                + (1.0 - self.alpha) * dense_scores.get(doc_id, 0.0)
            )
            for doc_id in all_docs
        }

        rrf_scores = {doc_id: 0.0 for doc_id in all_docs}
        for results in results_by_retriever.values():
            for doc_id, rank in results_to_rank_dict(results).items():
                rrf_scores[doc_id] += 1.0 / (self.k + rank)

        normalized_rrf = self.rrf_normalizer.normalize(rrf_scores)
        final_scores = {
            doc_id: (
                self.beta * normalized_rrf.get(doc_id, 0.0)
                + (1.0 - self.beta) * weighted_scores.get(doc_id, 0.0)
            )
            for doc_id in all_docs
        }

        return sort_and_truncate(final_scores, top_k)
