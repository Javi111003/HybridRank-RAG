"""
Tests unitarios para estrategias de fusión de HybridRank RAG.

Estos tests validan la lógica de fusión sin depender de recuperadores reales
(BM25, Dense) para evitar dependencias externas como sentence-transformers.
"""

import pytest
from typing import Dict, List, Tuple


# ===== Imports directos de módulos de fusión =====

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar módulos del paquete fusion directamente
from src.retriever.fusion.base import FusionStrategy
from src.retriever.fusion.utils import (
    results_to_score_dict,
    results_to_rank_dict,
    get_all_doc_ids,
    sort_and_truncate,
    fill_missing_scores,
)
from src.retriever.fusion.normalization import (
    MinMaxNormalizer,
    ZScoreNormalizer,
    SumNormalizer,
    IdentityNormalizer,
    get_normalizer,
)
from src.retriever.fusion.strategies import (
    ReciprocalRankFusion,
    BordaFusion,
    CombSUMFusion,
    CombMNZFusion,
    WeightedScoreFusion,
)
from src.retriever.fusion.hybrid_rank_fusion import HybridRankFusion
from src.retriever.fusion.registry import get_fusion_strategy


# ===== Tests de Utilidades =====

class TestUtils:

    def test_results_to_score_dict(self):
        results = [("doc1", 15.2), ("doc2", 12.1), ("doc3", 8.5)]
        score_dict = results_to_score_dict(results)
        assert score_dict == {"doc1": 15.2, "doc2": 12.1, "doc3": 8.5}

    def test_results_to_rank_dict(self):
        results = [("doc1", 15.2), ("doc2", 12.1), ("doc3", 8.5)]
        rank_dict = results_to_rank_dict(results)
        assert rank_dict == {"doc1": 1, "doc2": 2, "doc3": 3}

    def test_get_all_doc_ids(self):
        results_by_retriever = {
            "bm25": [("doc1", 15.2), ("doc2", 12.1)],
            "dense": [("doc3", 0.92), ("doc1", 0.85)]
        }
        all_ids = get_all_doc_ids(results_by_retriever)
        assert all_ids == {"doc1", "doc2", "doc3"}

    def test_sort_and_truncate(self):
        scores = {"doc1": 0.8, "doc2": 0.5, "doc3": 0.9}
        sorted_results = sort_and_truncate(scores, top_k=2)
        assert sorted_results == [("doc3", 0.9), ("doc1", 0.8)]

    def test_fill_missing_scores(self):
        score_dict = {"doc1": 0.8, "doc2": 0.5}
        all_docs = {"doc1", "doc2", "doc3"}
        filled = fill_missing_scores(score_dict, all_docs, default=0.0)
        assert filled == {"doc1": 0.8, "doc2": 0.5, "doc3": 0.0}


# ===== Tests de Normalizadores =====

class TestNormalizers:

    def test_minmax_normal_case(self):
        normalizer = MinMaxNormalizer()
        scores = {"doc1": 10.0, "doc2": 5.0, "doc3": 15.0}
        normalized = normalizer.normalize(scores)
        assert normalized == {"doc1": 0.5, "doc2": 0.0, "doc3": 1.0}

    def test_minmax_constant_scores(self):
        normalizer = MinMaxNormalizer()
        scores = {"doc1": 5.0, "doc2": 5.0, "doc3": 5.0}
        normalized = normalizer.normalize(scores)
        assert all(v == 1.0 for v in normalized.values())

    def test_minmax_empty(self):
        normalizer = MinMaxNormalizer()
        normalized = normalizer.normalize({})
        assert normalized == {}

    def test_zscore_normal_case(self):
        normalizer = ZScoreNormalizer()
        scores = {"doc1": 10.0, "doc2": 5.0, "doc3": 15.0}
        normalized = normalizer.normalize(scores)
        mean = 10.0
        assert abs(normalized["doc1"] - 0.0) < 0.01
        assert normalized["doc2"] < 0
        assert normalized["doc3"] > 0

    def test_zscore_constant_scores(self):
        normalizer = ZScoreNormalizer()
        scores = {"doc1": 5.0, "doc2": 5.0, "doc3": 5.0}
        normalized = normalizer.normalize(scores)
        assert all(v == 0.0 for v in normalized.values())

    def test_sum_normalizer(self):
        normalizer = SumNormalizer()
        scores = {"doc1": 10.0, "doc2": 20.0, "doc3": 30.0}
        normalized = normalizer.normalize(scores)
        assert abs(sum(normalized.values()) - 1.0) < 0.0001
        assert abs(normalized["doc1"] - 1/6) < 0.0001
        assert abs(normalized["doc2"] - 2/6) < 0.0001
        assert abs(normalized["doc3"] - 3/6) < 0.0001

    def test_identity(self):
        normalizer = IdentityNormalizer()
        scores = {"doc1": 15.2, "doc2": -3.5, "doc3": 100.0}
        normalized = normalizer.normalize(scores)
        assert normalized == scores

    def test_get_normalizer(self):
        assert isinstance(get_normalizer("minmax"), MinMaxNormalizer)
        assert isinstance(get_normalizer("zscore"), ZScoreNormalizer)
        assert isinstance(get_normalizer("sum"), SumNormalizer)
        assert isinstance(get_normalizer("identity"), IdentityNormalizer)

        with pytest.raises(ValueError):
            get_normalizer("unknown")


# ===== Tests de ReciprocalRankFusion =====

class TestReciprocalRankFusion:

    def test_basic_fusion(self):
        strategy = ReciprocalRankFusion(k=60)
        results_by_retriever = {
            "bm25": [("doc1", 15.2), ("doc2", 12.1), ("doc3", 8.5)],
            "dense": [("doc3", 0.92), ("doc1", 0.85), ("doc4", 0.78)]
        }
        fused = strategy.fuse(results_by_retriever, top_k=3)

        # doc1: 1/(60+1) + 1/(60+2) ≈ 0.0164 + 0.0161 = 0.0325
        # doc3: 1/(60+3) + 1/(60+1) ≈ 0.0159 + 0.0164 = 0.0323
        assert len(fused) == 3
        assert fused[0][0] == "doc1"  # Mayor score RRF

    def test_single_retriever(self):
        strategy = ReciprocalRankFusion(k=60)
        results_by_retriever = {
            "bm25": [("doc1", 15.2), ("doc2", 12.1)]
        }
        fused = strategy.fuse(results_by_retriever, top_k=2)
        assert fused[0][0] == "doc1"
        assert fused[1][0] == "doc2"

    def test_empty_results(self):
        strategy = ReciprocalRankFusion(k=60)
        fused = strategy.fuse({}, top_k=10)
        assert fused == []

    def test_different_k_values(self):
        strategy_k10 = ReciprocalRankFusion(k=10)
        strategy_k100 = ReciprocalRankFusion(k=100)

        results_by_retriever = {
            "bm25": [("doc1", 15.2), ("doc2", 12.1)],
            "dense": [("doc2", 0.92), ("doc1", 0.85)]
        }

        fused_k10 = strategy_k10.fuse(results_by_retriever, top_k=2)
        fused_k100 = strategy_k100.fuse(results_by_retriever, top_k=2)

        assert fused_k10[0][0] == fused_k100[0][0]
        assert fused_k10[0][1] != fused_k100[0][1]


# ===== Tests de BordaFusion =====

class TestBordaFusion:

    def test_basic_fusion(self):
        strategy = BordaFusion()
        results_by_retriever = {
            "bm25": [("doc1", 15.2), ("doc2", 12.1), ("doc3", 8.5)],
            "dense": [("doc3", 0.92), ("doc1", 0.85)]
        }
        fused = strategy.fuse(results_by_retriever, top_k=3)

        assert len(fused) == 3
        assert fused[0][0] == "doc1"
        assert fused[1][0] == "doc3"
        assert fused[2][0] == "doc2"

    def test_different_list_sizes(self):
        strategy = BordaFusion()
        results_by_retriever = {
            "bm25": [("doc1", 15.2), ("doc2", 12.1), ("doc3", 8.5), ("doc4", 5.0)],
            "dense": [("doc1", 0.92), ("doc3", 0.85)]
        }
        fused = strategy.fuse(results_by_retriever, top_k=4)
        assert fused[0][0] == "doc1"


# ===== Tests de CombSUMFusion =====

class TestCombSUMFusion:

    def test_with_minmax(self):
        strategy = CombSUMFusion(normalizer="minmax")
        results_by_retriever = {
            "bm25": [("doc1", 10.0), ("doc2", 5.0)],
            "dense": [("doc1", 0.8), ("doc2", 0.9)]
        }
        fused = strategy.fuse(results_by_retriever, top_k=2)

        assert len(fused) == 2
        scores = {doc_id: score for doc_id, score in fused}
        assert abs(scores["doc1"] - 1.0) < 0.01
        assert abs(scores["doc2"] - 1.0) < 0.01

    def test_partial_overlap(self):
        strategy = CombSUMFusion(normalizer="minmax")
        results_by_retriever = {
            "bm25": [("doc1", 10.0), ("doc2", 5.0)],
            "dense": [("doc3", 0.9), ("doc1", 0.8)]
        }
        fused = strategy.fuse(results_by_retriever, top_k=3)
        assert len(fused) == 3


# ===== Tests de CombMNZFusion =====

class TestCombMNZFusion:

    def test_penalizes_single_retriever_docs(self):
        strategy = CombMNZFusion(normalizer="minmax")
        results_by_retriever = {
            "bm25": [("doc1", 10.0), ("doc2", 5.0)],
            "dense": [("doc1", 0.9), ("doc3", 0.8)]
        }
        fused = strategy.fuse(results_by_retriever, top_k=3)
        assert fused[0][0] == "doc1"

    def test_with_full_overlap(self):
        strategy = CombMNZFusion(normalizer="minmax")
        results_by_retriever = {
            "bm25": [("doc1", 10.0), ("doc2", 5.0)],
            "dense": [("doc1", 0.9), ("doc2", 0.8)]
        }
        fused = strategy.fuse(results_by_retriever, top_k=2)
        assert len(fused) == 2


# ===== Tests de WeightedScoreFusion =====

class TestWeightedScoreFusion:

    def test_alpha_0_5(self):
        strategy = WeightedScoreFusion(alpha=0.5, sparse_key="bm25", dense_key="dense")
        results_by_retriever = {
            "bm25": [("doc1", 10.0), ("doc2", 5.0)],
            "dense": [("doc1", 0.8), ("doc2", 0.9)]
        }
        fused = strategy.fuse(results_by_retriever, top_k=2)
        assert len(fused) == 2

    def test_alpha_0_7(self):
        strategy = WeightedScoreFusion(alpha=0.7, sparse_key="bm25", dense_key="dense")
        results_by_retriever = {
            "bm25": [("doc1", 10.0), ("doc2", 5.0)],
            "dense": [("doc2", 0.9), ("doc1", 0.8)]
        }
        fused = strategy.fuse(results_by_retriever, top_k=2)
        assert fused[0][0] == "doc1"

    def test_alpha_0_3(self):
        strategy = WeightedScoreFusion(alpha=0.3, sparse_key="bm25", dense_key="dense")
        results_by_retriever = {
            "bm25": [("doc1", 10.0), ("doc2", 5.0)],
            "dense": [("doc2", 0.9), ("doc1", 0.8)]
        }
        fused = strategy.fuse(results_by_retriever, top_k=2)
        assert fused[0][0] == "doc2"

    def test_missing_in_one_retriever(self):
        strategy = WeightedScoreFusion(alpha=0.5, sparse_key="bm25", dense_key="dense")
        results_by_retriever = {
            "bm25": [("doc1", 10.0), ("doc2", 5.0)],
            "dense": [("doc1", 0.9), ("doc3", 0.8)]
        }
        fused = strategy.fuse(results_by_retriever, top_k=3)
        assert len(fused) == 3

    def test_missing_retriever_raises_error(self):
        strategy = WeightedScoreFusion(alpha=0.5, sparse_key="bm25", dense_key="dense")
        results_by_retriever = {
            "bm25": [("doc1", 10.0)]
        }

        with pytest.raises(ValueError, match="dense.*no encontrado"):
            strategy.fuse(results_by_retriever, top_k=1)


# ===== Tests de HybridRankFusion =====

class TestHybridRankFusion:

    def test_default_params(self):
        strategy = HybridRankFusion(alpha=0.5, beta=0.5)
        results_by_retriever = {
            "bm25": [("doc1", 10.0), ("doc2", 5.0)],
            "dense": [("doc1", 0.9), ("doc2", 0.8)]
        }
        fused = strategy.fuse(results_by_retriever, top_k=2)
        assert len(fused) == 2

    def test_beta_variations(self):
        strategy_beta_high = HybridRankFusion(alpha=0.5, beta=0.9)
        strategy_beta_low = HybridRankFusion(alpha=0.5, beta=0.1)

        results_by_retriever = {
            "bm25": [("doc1", 10.0), ("doc2", 5.0)],
            "dense": [("doc2", 0.9), ("doc1", 0.8)]
        }

        fused_high = strategy_beta_high.fuse(results_by_retriever, top_k=2)
        fused_low = strategy_beta_low.fuse(results_by_retriever, top_k=2)

        assert len(fused_high) == 2
        assert len(fused_low) == 2

    def test_alpha_variations(self):
        strategy_alpha_high = HybridRankFusion(alpha=0.9, beta=0.5)
        strategy_alpha_low = HybridRankFusion(alpha=0.1, beta=0.5)

        results_by_retriever = {
            "bm25": [("doc1", 10.0), ("doc2", 5.0)],
            "dense": [("doc2", 0.9), ("doc1", 0.8)]
        }

        fused_high = strategy_alpha_high.fuse(results_by_retriever, top_k=2)
        fused_low = strategy_alpha_low.fuse(results_by_retriever, top_k=2)

        assert len(fused_high) == 2
        assert len(fused_low) == 2

    def test_combined_variations(self):
        strategy = HybridRankFusion(alpha=0.7, beta=0.3, k=100)
        results_by_retriever = {
            "bm25": [("doc1", 10.0), ("doc2", 5.0), ("doc3", 2.0)],
            "dense": [("doc3", 0.95), ("doc2", 0.85), ("doc1", 0.75)]
        }
        fused = strategy.fuse(results_by_retriever, top_k=3)
        assert len(fused) == 3

    def test_parameter_validation(self):
        with pytest.raises(ValueError):
            HybridRankFusion(alpha=1.5)
        with pytest.raises(ValueError):
            HybridRankFusion(beta=-0.1)
        with pytest.raises(ValueError):
            HybridRankFusion(k=0)


# ===== Tests de Registry =====

class TestRegistry:

    def test_get_all_strategies(self):
        assert isinstance(get_fusion_strategy("rrf"), ReciprocalRankFusion)
        assert isinstance(get_fusion_strategy("borda"), BordaFusion)
        assert isinstance(get_fusion_strategy("combsum"), CombSUMFusion)
        assert isinstance(get_fusion_strategy("combmnz"), CombMNZFusion)
        assert isinstance(get_fusion_strategy("weighted"), WeightedScoreFusion)
        assert isinstance(get_fusion_strategy("hybridrank"), HybridRankFusion)

    def test_with_params(self):
        strategy = get_fusion_strategy("rrf", k=100)
        assert isinstance(strategy, ReciprocalRankFusion)
        assert strategy.k == 100

        strategy = get_fusion_strategy("weighted", alpha=0.7)
        assert isinstance(strategy, WeightedScoreFusion)
        assert strategy.alpha == 0.7

        strategy = get_fusion_strategy("hybridrank", alpha=0.6, beta=0.4, k=80)
        assert isinstance(strategy, HybridRankFusion)
        assert strategy.alpha == 0.6
        assert strategy.beta == 0.4
        assert strategy.k == 80

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="desconocida"):
            get_fusion_strategy("unknown")
