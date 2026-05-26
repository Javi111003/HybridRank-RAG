"""Política adaptativa de fusión: aprende la mejor configuración por tipo de consulta."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.retriever.fusion.registry import get_fusion_strategy
from src.retriever.fusion.base import FusionStrategy
from src.retriever.metrics import RecallAtK, PrecisionAtK, F1AtK, MRR, MAP, NDCG

logger = logging.getLogger(__name__)

OBJECTIVE_WEIGHTS = {"ndcg": 0.4, "recall": 0.3, "map": 0.2, "f1": 0.1}
OBJECTIVE_K = 10


@dataclass
class FusionConfig:
    """Una configuración específica de estrategia de fusión."""

    strategy_name: str
    params: Dict[str, Any]
    candidate_k: int = 50

    def build_strategy(self) -> FusionStrategy:
        return get_fusion_strategy(self.strategy_name, **self.params)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "params": self.params,
            "candidate_k": self.candidate_k,
        }

    def params_json(self) -> str:
        return json.dumps(self.params, sort_keys=True)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FusionConfig):
            return NotImplemented
        return (
            self.strategy_name == other.strategy_name
            and self.params == other.params
            and self.candidate_k == other.candidate_k
        )

    def __hash__(self) -> int:
        return hash((self.strategy_name, self.params_json(), self.candidate_k))


GLOBAL_FALLBACK = FusionConfig(
    strategy_name="hybridrank",
    params={
        "alpha": 0.7,
        "beta": 0.2,
        "k": 10,
        "normalizer": "minmax",
        "rrf_normalizer": "minmax",
    },
    candidate_k=50,
)


def compute_objective(metrics_dict: Dict[str, float]) -> float:
    """Calcula el objetivo compuesto: 0.4*nDCG + 0.3*Recall + 0.2*MAP + 0.1*F1."""
    return (
        OBJECTIVE_WEIGHTS["ndcg"] * metrics_dict.get("ndcg", 0.0)
        + OBJECTIVE_WEIGHTS["recall"] * metrics_dict.get("recall", 0.0)
        + OBJECTIVE_WEIGHTS["map"] * metrics_dict.get("map", 0.0)
        + OBJECTIVE_WEIGHTS["f1"] * metrics_dict.get("f1", 0.0)
    )


def compute_metrics_for_query(
    fused_results: List[Tuple[str, float]],
    relevant_docs: List[str],
    metrics: List[Any],
    k: int,
) -> Dict[str, float]:
    """Computa todas las métricas para una query fusionada."""
    result = {}
    for metric in metrics:
        score = metric.compute(fused_results, relevant_docs, k=k)["score"]
        col_name = metric.name.lower().replace("@k", "")
        result[col_name] = score
    return result


def build_search_space() -> List[FusionConfig]:
    """Espacio de búsqueda completo: 300 HybridRank + 54 Weighted = 354 configs."""
    configs: List[FusionConfig] = []

    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            for rrf_k in [10, 20, 40, 60, 100]:
                for normalizer in ["minmax", "zscore"]:
                    configs.append(
                        FusionConfig(
                            strategy_name="hybridrank",
                            params={
                                "alpha": alpha,
                                "beta": beta,
                                "k": rrf_k,
                                "normalizer": normalizer,
                                "rrf_normalizer": normalizer,
                            },
                            candidate_k=50,
                        )
                    )

    for alpha in [round(x * 0.1, 1) for x in range(1, 10)]:
        for normalizer in ["minmax", "zscore"]:
            for candidate_k in [20, 50, 100]:
                configs.append(
                    FusionConfig(
                        strategy_name="weighted",
                        params={"alpha": alpha, "normalizer": normalizer},
                        candidate_k=candidate_k,
                    )
                )

    return configs


class AdaptiveFusionPolicy:
    """
    Aprende la mejor configuración de fusión por tipo de consulta desde datos de entrenamiento.

    fit() evalúa todas las configs del espacio de búsqueda sobre queries de train
    agrupadas por query_type, seleccionando la mejor config por tipo.

    select_config() devuelve la config aprendida para un tipo dado, o el fallback
    global si el tipo no tiene suficientes ejemplos de entrenamiento.
    """

    def __init__(
        self,
        search_space: Optional[List[FusionConfig]] = None,
        global_fallback: Optional[FusionConfig] = None,
        min_train_examples_per_type: int = 3,
    ):
        self._search_space = search_space if search_space is not None else build_search_space()
        self._fallback = global_fallback or GLOBAL_FALLBACK
        self._min_examples = min_train_examples_per_type
        self._best_config_by_type: Dict[str, FusionConfig] = {}
        self._train_counts_by_type: Dict[str, int] = {}
        self._fallback_used_by_type: Dict[str, bool] = {}
        self._fitted = False

    def fit(
        self,
        train_queries: List[Dict[str, Any]],
        cache: Dict[int, Dict[str, Dict[str, List[Tuple[str, float]]]]],
        query_type_assignments: Dict[str, str],
    ) -> None:
        """
        Aprende la mejor config por query_type desde train.

        Args:
            train_queries: Lista de dicts con query_id, relevant_docs, etc.
            cache: {candidate_k: {query_id: {"bm25": [...], "dense": [...]}}}
            query_type_assignments: Mapeo query_id → query_type asignado.
        """
        metrics = [RecallAtK(), PrecisionAtK(), F1AtK(), MRR(), MAP(), NDCG()]

        queries_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for q in train_queries:
            qid = q["query_id"]
            assigned_type = query_type_assignments.get(qid, "semantica")
            queries_by_type[assigned_type].append(q)

        self._train_counts_by_type = {t: len(qs) for t, qs in queries_by_type.items()}
        self._best_config_by_type = {}
        self._fallback_used_by_type = {}

        for query_type, type_queries in queries_by_type.items():
            if len(type_queries) < self._min_examples:
                self._fallback_used_by_type[query_type] = True
                logger.info(
                    "Tipo '%s' tiene %d queries (< %d). Usando fallback.",
                    query_type, len(type_queries), self._min_examples,
                )
                continue

            best_objective = -1.0
            best_config = None

            for config in self._search_space:
                ck = config.candidate_k
                if ck not in cache:
                    continue

                objectives = []
                for q in type_queries:
                    qid = q["query_id"]
                    if qid not in cache[ck]:
                        continue
                    results_by_retriever = cache[ck][qid]
                    strategy = config.build_strategy()
                    fused = strategy.fuse(results_by_retriever, top_k=20)
                    m = compute_metrics_for_query(
                        fused, q["relevant_docs"], metrics, OBJECTIVE_K
                    )
                    objectives.append(compute_objective(m))

                if not objectives:
                    continue

                mean_obj = float(np.mean(objectives))
                if mean_obj > best_objective:
                    best_objective = mean_obj
                    best_config = config

            if best_config is not None:
                self._best_config_by_type[query_type] = best_config
                self._fallback_used_by_type[query_type] = False
            else:
                self._fallback_used_by_type[query_type] = True

        self._fitted = True

    def select_config(
        self,
        query_type: str,
        confidence: float,
        available_train_counts: Optional[Dict[str, int]] = None,
    ) -> FusionConfig:
        """
        Selecciona la config para un tipo de consulta.

        Usa fallback si:
        - La política no ha sido entrenada
        - El tipo no tiene config aprendida
        - La confianza de clasificación es muy baja (< 0.5)
        """
        if not self._fitted:
            return self._fallback

        if confidence < 0.5:
            return self._fallback

        return self._best_config_by_type.get(query_type, self._fallback)

    @property
    def best_config_by_type(self) -> Dict[str, FusionConfig]:
        return dict(self._best_config_by_type)

    @property
    def train_counts_by_type(self) -> Dict[str, int]:
        return dict(self._train_counts_by_type)

    @property
    def fallback_used_by_type(self) -> Dict[str, bool]:
        return dict(self._fallback_used_by_type)

    def is_fallback(self, query_type: str) -> bool:
        return self._fallback_used_by_type.get(query_type, True)
