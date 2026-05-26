"""Tests para AdaptiveHybridRank orquestador."""

import pytest
from unittest.mock import MagicMock

from src.adaptive.adaptive_hybridrank import AdaptiveHybridRank
from src.adaptive.classification import (
    RuleBasedQueryClassifier,
    OracleQueryClassifier,
    QueryClassification,
)
from src.adaptive.policy import AdaptiveFusionPolicy, FusionConfig, GLOBAL_FALLBACK
from src.adaptive.query_signals import QuerySignalExtractor


class TestAdaptiveHybridRank:
    def _make_mock_retriever(self, results):
        mock = MagicMock()
        mock.retrieve.return_value = results
        return mock

    def test_retrieve_returns_valid_results(self):
        bm25 = self._make_mock_retriever(
            [("d1", 10.0), ("d2", 8.0), ("d3", 5.0)]
        )
        dense = self._make_mock_retriever(
            [("d1", 0.9), ("d2", 0.85), ("d4", 0.7)]
        )
        clf = RuleBasedQueryClassifier()
        policy = AdaptiveFusionPolicy()

        ahr = AdaptiveHybridRank(
            retrievers={"bm25": bm25, "dense": dense},
            classifier=clf,
            policy=policy,
        )

        results = ahr.retrieve("Decreto-Ley 114 de 2025 sobre impuestos", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)

    def test_retrieve_with_metadata(self):
        bm25 = self._make_mock_retriever([("d1", 10.0), ("d2", 8.0)])
        dense = self._make_mock_retriever([("d1", 0.9), ("d3", 0.8)])
        clf = RuleBasedQueryClassifier()
        policy = AdaptiveFusionPolicy()

        ahr = AdaptiveHybridRank(
            retrievers={"bm25": bm25, "dense": dense},
            classifier=clf,
            policy=policy,
        )

        meta = ahr.retrieve_with_metadata("Decreto 5 de 2024", top_k=2)
        assert "results" in meta
        assert "query_type" in meta
        assert "confidence" in meta
        assert "config" in meta
        assert "signals" in meta
        assert "fallback_used" in meta
        assert meta["query_type"] == "referencia_exacta"

    def test_uses_oracle_classifier_with_id(self):
        bm25 = self._make_mock_retriever([("d1", 10.0)])
        dense = self._make_mock_retriever([("d1", 0.9)])
        oracle = OracleQueryClassifier({"q1": "multi_hop", "q2": "semantica"})
        policy = AdaptiveFusionPolicy()

        ahr = AdaptiveHybridRank(
            retrievers={"bm25": bm25, "dense": dense},
            classifier=oracle,
            policy=policy,
        )

        meta = ahr.retrieve_with_metadata("cualquier query", query_id="q1", top_k=1)
        assert meta["query_type"] == "multi_hop"

    def test_unfitted_policy_uses_fallback(self):
        bm25 = self._make_mock_retriever([("d1", 10.0), ("d2", 8.0)])
        dense = self._make_mock_retriever([("d1", 0.9), ("d2", 0.85)])
        clf = RuleBasedQueryClassifier()
        policy = AdaptiveFusionPolicy()

        ahr = AdaptiveHybridRank(
            retrievers={"bm25": bm25, "dense": dense},
            classifier=clf,
            policy=policy,
        )

        meta = ahr.retrieve_with_metadata("protección ambiental", top_k=2)
        assert meta["fallback_used"] is True
        assert meta["config"] == GLOBAL_FALLBACK

    def test_handles_empty_retrieval(self):
        bm25 = self._make_mock_retriever([])
        dense = self._make_mock_retriever([])
        clf = RuleBasedQueryClassifier()
        policy = AdaptiveFusionPolicy()

        ahr = AdaptiveHybridRank(
            retrievers={"bm25": bm25, "dense": dense},
            classifier=clf,
            policy=policy,
        )

        results = ahr.retrieve("test", top_k=5)
        assert results == []

    def test_candidate_k_passed_to_retrievers(self):
        bm25 = self._make_mock_retriever([("d1", 10.0)])
        dense = self._make_mock_retriever([("d1", 0.9)])
        clf = RuleBasedQueryClassifier()
        policy = AdaptiveFusionPolicy()

        ahr = AdaptiveHybridRank(
            retrievers={"bm25": bm25, "dense": dense},
            classifier=clf,
            policy=policy,
            candidate_k_default=100,
        )

        ahr.retrieve("test", top_k=5)
        bm25.retrieve.assert_called_once_with("test", 100)
        dense.retrieve.assert_called_once_with("test", 100)
