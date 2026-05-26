"""Tests para AdaptiveFusionPolicy."""

import pytest
import numpy as np

from src.adaptive.policy import (
    AdaptiveFusionPolicy,
    FusionConfig,
    GLOBAL_FALLBACK,
    compute_objective,
    compute_metrics_for_query,
    build_search_space,
)
from src.retriever.metrics import RecallAtK, PrecisionAtK, F1AtK, MRR, MAP, NDCG


class TestFusionConfig:
    def test_build_strategy_hybridrank(self):
        config = FusionConfig(
            strategy_name="hybridrank",
            params={
                "alpha": 0.7, "beta": 0.2, "k": 10,
                "normalizer": "minmax", "rrf_normalizer": "minmax",
            },
        )
        strategy = config.build_strategy()
        assert strategy.alpha == 0.7
        assert strategy.beta == 0.2

    def test_build_strategy_weighted(self):
        config = FusionConfig(
            strategy_name="weighted",
            params={"alpha": 0.6, "normalizer": "minmax"},
        )
        strategy = config.build_strategy()
        assert hasattr(strategy, "fuse")

    def test_to_dict(self):
        config = FusionConfig("rrf", {"k": 60}, 50)
        d = config.to_dict()
        assert d["strategy_name"] == "rrf"
        assert d["params"]["k"] == 60
        assert d["candidate_k"] == 50

    def test_equality(self):
        c1 = FusionConfig("hybridrank", {"alpha": 0.7}, 50)
        c2 = FusionConfig("hybridrank", {"alpha": 0.7}, 50)
        c3 = FusionConfig("hybridrank", {"alpha": 0.8}, 50)
        assert c1 == c2
        assert c1 != c3

    def test_hash(self):
        c1 = FusionConfig("hybridrank", {"alpha": 0.7}, 50)
        c2 = FusionConfig("hybridrank", {"alpha": 0.7}, 50)
        assert hash(c1) == hash(c2)


class TestComputeObjective:
    def test_perfect_scores(self):
        metrics = {"ndcg": 1.0, "recall": 1.0, "map": 1.0, "f1": 1.0}
        assert compute_objective(metrics) == pytest.approx(1.0)

    def test_zero_scores(self):
        metrics = {"ndcg": 0.0, "recall": 0.0, "map": 0.0, "f1": 0.0}
        assert compute_objective(metrics) == 0.0

    def test_weighted_combination(self):
        metrics = {"ndcg": 0.5, "recall": 0.5, "map": 0.5, "f1": 0.5}
        assert compute_objective(metrics) == pytest.approx(0.5)

    def test_missing_keys_default_zero(self):
        metrics = {"ndcg": 1.0}
        assert compute_objective(metrics) == pytest.approx(0.4)


class TestBuildSearchSpace:
    def test_nonempty(self):
        space = build_search_space()
        assert len(space) > 0

    def test_contains_hybridrank_and_weighted(self):
        space = build_search_space()
        names = {c.strategy_name for c in space}
        assert "hybridrank" in names
        assert "weighted" in names

    def test_expected_size(self):
        space = build_search_space()
        assert len(space) == 354  # 300 hybridrank + 54 weighted


class TestAdaptiveFusionPolicy:
    def _make_cache(self):
        """Cache sintético con resultados de retrieval para 5 queries."""
        cache = {50: {}}
        for i in range(1, 6):
            qid = f"q{i}"
            cache[50][qid] = {
                "bm25": [(f"d{j}", 10.0 - j) for j in range(1, 11)],
                "dense": [(f"d{j}", 0.9 - j * 0.05) for j in range(1, 11)],
            }
        return cache

    def _make_train_queries(self, n=5, query_type="semantica"):
        """Queries de train sintéticas con docs relevantes."""
        queries = []
        for i in range(1, n + 1):
            queries.append({
                "query_id": f"q{i}",
                "query": f"query de prueba {i}",
                "query_type": query_type,
                "relevant_docs": [f"d{i}", f"d{i+1}"],
            })
        return queries

    def test_unfitted_returns_fallback(self):
        policy = AdaptiveFusionPolicy()
        config = policy.select_config("semantica", 0.9)
        assert config == GLOBAL_FALLBACK

    def test_fit_with_sufficient_examples(self):
        small_space = [
            GLOBAL_FALLBACK,
            FusionConfig("weighted", {"alpha": 0.5, "normalizer": "minmax"}, 50),
        ]
        policy = AdaptiveFusionPolicy(search_space=small_space, min_train_examples_per_type=3)

        cache = self._make_cache()
        train = self._make_train_queries(4, "semantica")
        assignments = {q["query_id"]: "semantica" for q in train}

        policy.fit(train, cache, assignments)
        config = policy.select_config("semantica", 0.9)
        assert config in small_space

    def test_insufficient_examples_returns_fallback(self):
        small_space = [
            GLOBAL_FALLBACK,
            FusionConfig("weighted", {"alpha": 0.5, "normalizer": "minmax"}, 50),
        ]
        policy = AdaptiveFusionPolicy(search_space=small_space, min_train_examples_per_type=3)

        cache = self._make_cache()
        train = self._make_train_queries(2, "temporal_historica")
        assignments = {q["query_id"]: "temporal_historica" for q in train}

        policy.fit(train, cache, assignments)
        assert policy.is_fallback("temporal_historica")
        config = policy.select_config("temporal_historica", 0.9)
        assert config == GLOBAL_FALLBACK

    def test_low_confidence_returns_fallback(self):
        small_space = [GLOBAL_FALLBACK]
        policy = AdaptiveFusionPolicy(search_space=small_space)

        cache = self._make_cache()
        train = self._make_train_queries(4, "semantica")
        assignments = {q["query_id"]: "semantica" for q in train}
        policy.fit(train, cache, assignments)

        config = policy.select_config("semantica", 0.3)
        assert config == GLOBAL_FALLBACK

    def test_unknown_type_returns_fallback(self):
        policy = AdaptiveFusionPolicy()
        cache = self._make_cache()
        train = self._make_train_queries(4, "semantica")
        assignments = {q["query_id"]: "semantica" for q in train}
        policy.fit(train, cache, assignments)

        config = policy.select_config("tipo_desconocido", 0.9)
        assert config == GLOBAL_FALLBACK

    def test_train_counts_tracked(self):
        small_space = [GLOBAL_FALLBACK]
        policy = AdaptiveFusionPolicy(search_space=small_space)

        cache = self._make_cache()
        train = self._make_train_queries(4, "semantica")
        assignments = {q["query_id"]: "semantica" for q in train}
        policy.fit(train, cache, assignments)

        assert policy.train_counts_by_type["semantica"] == 4
