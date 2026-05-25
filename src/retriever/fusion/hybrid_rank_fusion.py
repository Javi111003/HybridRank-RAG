from typing import Dict, List, Tuple

from .base import FusionStrategy
from .normalization import get_normalizer
from .utils import (
    get_all_doc_ids,
    results_to_rank_dict,
    results_to_score_dict,
    sort_and_truncate,
)


class HybridRankFusion(FusionStrategy):
    """
    HybridRank Fusion - Primera propuesta experimental de HybridRank RAG.

    Familia: Híbrida (ranking + scores).

    Combina dos señales complementarias:
    1. RRF (Reciprocal Rank Fusion): robusto ante escalas incompatibles,
       basado solo en posiciones.
    2. Weighted Score Fusion: aprovecha magnitudes de scores normalizados,
       con control del balance sparse/dense.

    Arquitectura de dos niveles:

    Nivel 1 - Score ponderado:
        score_weighted(d) = alpha * norm_sparse(d) + (1 - alpha) * norm_dense(d)

    Nivel 2 - RRF:
        score_rrf(d) = sum_s [ 1 / (k + rank_s(d)) ]

    Fusión final:
        score_final(d) = beta * norm_rrf(d) + (1 - beta) * score_weighted(d)

    Parámetros de Control:
        - alpha ∈ [0, 1]: Balance sparse vs dense en weighted fusion.
            * alpha alto (0.7-0.9) → favorece BM25 (term matching exacto)
            * alpha bajo (0.1-0.3) → favorece Dense (semántica)
            * alpha = 0.5 → balance igual

        - beta ∈ [0, 1]: Balance RRF vs WeightedScore en fusión final.
            * beta alto (0.7-0.9) → favorece RRF (robustez, consenso)
            * beta bajo (0.1-0.3) → favorece scores (magnitudes)
            * beta = 0.5 → balance igual

        - k: Parámetro de suavizado RRF (default: 60).

    Casos especiales:
        - beta = 0 → equivale a WeightedScoreFusion
        - beta = 1 → equivale a RRF puro
        - alpha = 0.5, beta = 0.5 → máximo balance

    Ventajas:
        - Combina robustez de RRF con información de magnitudes
        - Control fino de balance sparse/dense y ranking/score
        - Dos niveles de normalización para estabilidad
        - Adaptable a diferentes tipos de queries

    Limitaciones:
        - Más parámetros que ajustar (alpha, beta, k)
        - Mayor complejidad computacional
        - Requiere dos recuperadores (sparse y dense)

    Normalización requerida: Sí (2 normalizadores).
    """

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
        """
        Args:
            alpha: Peso sparse en weighted fusion [0, 1]. Default: 0.7.
            beta: Peso RRF en fusión final [0, 1]. Default: 0.2.
            k: Parámetro de suavizado RRF. Default: 10.
            sparse_key: Nombre del recuperador sparse. Default: "bm25".
            dense_key: Nombre del recuperador dense. Default: "dense".
            normalizer: Normalizador para sparse/dense. Default: "minmax".
            rrf_normalizer: Normalizador para scores RRF. Default: "minmax".

        Raises:
            ValueError: Si alpha o beta están fuera de [0, 1], o k <= 0.
        """
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
        results_by_retriever: Dict[str, List[Tuple[str, float]]],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        if not results_by_retriever:
            return []

        # Validar que existen ambos recuperadores
        if self.sparse_key not in results_by_retriever:
            raise ValueError(
                f"Recuperador sparse '{self.sparse_key}' no encontrado. "
                f"Disponibles: {list(results_by_retriever.keys())}"
            )
        if self.dense_key not in results_by_retriever:
            raise ValueError(
                f"Recuperador dense '{self.dense_key}' no encontrado. "
                f"Disponibles: {list(results_by_retriever.keys())}"
            )

        all_docs = get_all_doc_ids(results_by_retriever)

        # ===== Nivel 1: Weighted Score Fusion =====
        sparse_results = results_by_retriever[self.sparse_key]
        dense_results = results_by_retriever[self.dense_key]

        sparse_scores = results_to_score_dict(sparse_results)
        dense_scores = results_to_score_dict(dense_results)

        sparse_normalized = self.normalizer.normalize(sparse_scores)
        dense_normalized = self.normalizer.normalize(dense_scores)

        weighted_scores: Dict[str, float] = {}
        for doc_id in all_docs:
            sparse_score = sparse_normalized.get(doc_id, 0.0)
            dense_score = dense_normalized.get(doc_id, 0.0)
            weighted_scores[doc_id] = (
                self.alpha * sparse_score + (1.0 - self.alpha) * dense_score
            )

        # ===== Nivel 2: RRF =====
        rrf_scores: Dict[str, float] = {doc_id: 0.0 for doc_id in all_docs}

        for retriever_name, results in results_by_retriever.items():
            ranks = results_to_rank_dict(results)
            for doc_id, rank in ranks.items():
                rrf_scores[doc_id] += 1.0 / (self.k + rank)

        # Normalizar scores RRF
        rrf_normalized = self.rrf_normalizer.normalize(rrf_scores)

        # ===== Fusión Final =====
        final_scores: Dict[str, float] = {}
        for doc_id in all_docs:
            rrf_score = rrf_normalized.get(doc_id, 0.0)
            weighted_score = weighted_scores.get(doc_id, 0.0)
            final_scores[doc_id] = (
                self.beta * rrf_score + (1.0 - self.beta) * weighted_score
            )

        return sort_and_truncate(final_scores, top_k)
