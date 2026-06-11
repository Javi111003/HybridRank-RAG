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


class ReciprocalRankFusion(FusionStrategy):
    """Rank-only fusion based on reciprocal positions."""

    def __init__(self, k: int = 60):
        if k <= 0:
            raise ValueError(f"k debe ser > 0, recibido: {k}")
        self.k = k

    def fuse(
        self,
        results_by_retriever: ResultsByRetriever,
        top_k: int,
    ) -> RetrievalResults:
        if not results_by_retriever:
            return []

        scores = {doc_id: 0.0 for doc_id in get_all_doc_ids(results_by_retriever)}
        for results in results_by_retriever.values():
            for doc_id, rank in results_to_rank_dict(results).items():
                scores[doc_id] += 1.0 / (self.k + rank)

        return sort_and_truncate(scores, top_k)


class BordaFusion(FusionStrategy):
    """Rank-only fusion using Borda count."""

    def fuse(
        self,
        results_by_retriever: ResultsByRetriever,
        top_k: int,
    ) -> RetrievalResults:
        if not results_by_retriever:
            return []

        scores: dict[str, float] = {}
        for results in results_by_retriever.values():
            list_size = len(results)
            for doc_id, rank in results_to_rank_dict(results).items():
                scores[doc_id] = scores.get(doc_id, 0.0) + list_size - rank + 1

        return sort_and_truncate(scores, top_k)


class CombSUMFusion(FusionStrategy):
    """Score fusion by summing normalized scores."""

    def __init__(self, normalizer: str = "minmax"):
        self.normalizer = get_normalizer(normalizer)

    def fuse(
        self,
        results_by_retriever: ResultsByRetriever,
        top_k: int,
    ) -> RetrievalResults:
        if not results_by_retriever:
            return []

        all_docs = get_all_doc_ids(results_by_retriever)
        scores = {doc_id: 0.0 for doc_id in all_docs}

        for results in results_by_retriever.values():
            normalized = _normalize(results, self.normalizer)
            for doc_id in all_docs:
                scores[doc_id] += normalized.get(doc_id, 0.0)

        return sort_and_truncate(scores, top_k)


class CombMNZFusion(FusionStrategy):
    """CombSUM weighted by the number of positive normalized signals."""

    def __init__(self, normalizer: str = "minmax"):
        self.normalizer = get_normalizer(normalizer)

    def fuse(
        self,
        results_by_retriever: ResultsByRetriever,
        top_k: int,
    ) -> RetrievalResults:
        if not results_by_retriever:
            return []

        all_docs = get_all_doc_ids(results_by_retriever)
        sums = {doc_id: 0.0 for doc_id in all_docs}
        counts = {doc_id: 0 for doc_id in all_docs}

        for results in results_by_retriever.values():
            normalized = _normalize(results, self.normalizer)
            for doc_id in all_docs:
                score = normalized.get(doc_id, 0.0)
                if score > 0.0:
                    sums[doc_id] += score
                    counts[doc_id] += 1

        scores = {doc_id: counts[doc_id] * sums[doc_id] for doc_id in all_docs}
        return sort_and_truncate(scores, top_k)


class WeightedScoreFusion(FusionStrategy):
    """Linear sparse/dense score fusion."""

    def __init__(
        self,
        alpha: float = 0.5,
        sparse_key: str = "bm25",
        dense_key: str = "dense",
        normalizer: str = "minmax",
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha debe estar en [0, 1], recibido: {alpha}")

        self.alpha = alpha
        self.sparse_key = sparse_key
        self.dense_key = dense_key
        self.normalizer = get_normalizer(normalizer)

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
        dense_results = require_results(
            results_by_retriever, self.dense_key, "dense"
        )

        sparse_scores = _normalize(sparse_results, self.normalizer)
        dense_scores = _normalize(dense_results, self.normalizer)

        scores = {
            doc_id: (
                self.alpha * sparse_scores.get(doc_id, 0.0)
                + (1.0 - self.alpha) * dense_scores.get(doc_id, 0.0)
            )
            for doc_id in get_all_doc_ids(results_by_retriever)
        }
        return sort_and_truncate(scores, top_k)
