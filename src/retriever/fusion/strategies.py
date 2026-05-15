from typing import Dict, List, Tuple

from .base import FusionStrategy
from .normalization import get_normalizer
from .utils import (
    fill_missing_scores,
    get_all_doc_ids,
    results_to_rank_dict,
    results_to_score_dict,
    sort_and_truncate,
)


class ReciprocalRankFusion(FusionStrategy):
    """
    Reciprocal Rank Fusion (RRF).

    Familia: Basada en rankings.

    Fusiona rankings usando solo las posiciones, ignorando los scores originales.
    Especialmente robusto ante escalas incompatibles entre recuperadores.

    Fórmula:
        score_rrf(d) = sum_s [ 1 / (k + rank_s(d)) ]

    donde:
        - rank_s(d) es la posición del documento d en el recuperador s (1-indexed)
        - k es un parámetro de suavizado (default: 60)

    Ventajas:
        - No requiere normalización de scores
        - Robusto ante escalas incompatibles
        - Simple y efectivo

    Limitaciones:
        - Pierde información sobre la magnitud de los scores
        - Todos los recuperadores tienen igual peso implícito
    """

    def __init__(self, k: int = 60):
        """
        Args:
            k: Parámetro de suavizado. Valores típicos: 60 (default), 100.
        """
        if k <= 0:
            raise ValueError(f"k debe ser > 0, recibido: {k}")
        self.k = k

    def fuse(
        self,
        results_by_retriever: Dict[str, List[Tuple[str, float]]],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        if not results_by_retriever:
            return []

        all_docs = get_all_doc_ids(results_by_retriever)
        rrf_scores: Dict[str, float] = {doc_id: 0.0 for doc_id in all_docs}

        for retriever_name, results in results_by_retriever.items():
            ranks = results_to_rank_dict(results)
            for doc_id, rank in ranks.items():
                rrf_scores[doc_id] += 1.0 / (self.k + rank)

        return sort_and_truncate(rrf_scores, top_k)


class BordaFusion(FusionStrategy):
    """
    Borda Count Fusion.

    Familia: Basada en rankings.

    Asigna puntos basados en la posición relativa de cada documento
    en cada lista. El documento en primera posición recibe N puntos,
    el segundo N-1, etc.

    Fórmula:
        score_borda(d) = sum_s [ N_s - rank_s(d) + 1 ]

    donde:
        - N_s es el número de documentos en la lista del recuperador s
        - rank_s(d) es la posición de d en s (1-indexed)

    Ventajas:
        - Simple e intuitivo
        - Considera el tamaño relativo de cada lista

    Limitaciones:
        - Sesgo hacia recuperadores que retornan más documentos
        - Ignora magnitudes de scores
    """

    def fuse(
        self,
        results_by_retriever: Dict[str, List[Tuple[str, float]]],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        if not results_by_retriever:
            return []

        all_docs = get_all_doc_ids(results_by_retriever)
        borda_scores: Dict[str, float] = {doc_id: 0.0 for doc_id in all_docs}

        for retriever_name, results in results_by_retriever.items():
            N = len(results)
            ranks = results_to_rank_dict(results)
            for doc_id, rank in ranks.items():
                borda_scores[doc_id] += N - rank + 1

        return sort_and_truncate(borda_scores, top_k)


class CombSUMFusion(FusionStrategy):
    """
    CombSUM Fusion.

    Familia: Basada en scores.

    Normaliza los scores de cada recuperador y luego los suma.
    Requiere normalización para evitar que un recuperador domine.

    Fórmula:
        score_combsum(d) = sum_s [ normalized_score_s(d) ]

    Ventajas:
        - Aprovecha la información de magnitud de scores
        - Flexible (permite elegir normalizador)

    Limitaciones:
        - Requiere normalización adecuada
        - Sensible a la elección del normalizador

    Normalización requerida: Sí (default: minmax).
    """

    def __init__(self, normalizer: str = "minmax"):
        """
        Args:
            normalizer: Nombre del normalizador: "minmax", "zscore", "sum", "identity".
        """
        self.normalizer = get_normalizer(normalizer)

    def fuse(
        self,
        results_by_retriever: Dict[str, List[Tuple[str, float]]],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        if not results_by_retriever:
            return []

        all_docs = get_all_doc_ids(results_by_retriever)
        combsum_scores: Dict[str, float] = {doc_id: 0.0 for doc_id in all_docs}

        for retriever_name, results in results_by_retriever.items():
            score_dict = results_to_score_dict(results)
            normalized = self.normalizer.normalize(score_dict)

            for doc_id in all_docs:
                combsum_scores[doc_id] += normalized.get(doc_id, 0.0)

        return sort_and_truncate(combsum_scores, top_k)


class CombMNZFusion(FusionStrategy):
    """
    CombMNZ (Combination of Multiple Normalizations with Zero handling).

    Familia: Basada en scores.

    Similar a CombSUM, pero multiplica la suma por el número de recuperadores
    que encontraron el documento. Favorece documentos que aparecen en múltiples
    recuperadores.

    Fórmula:
        score_combmnz(d) = count_nonzero(d) * sum_s [ normalized_score_s(d) ]

    donde count_nonzero(d) es el número de recuperadores que recuperaron d.

    Ventajas:
        - Favorece consenso entre recuperadores
        - Penaliza documentos que solo aparecen en un recuperador

    Limitaciones:
        - Puede ser demasiado conservador
        - Penaliza hallazgos únicos válidos

    Normalización requerida: Sí (default: minmax).
    """

    def __init__(self, normalizer: str = "minmax"):
        """
        Args:
            normalizer: Nombre del normalizador: "minmax", "zscore", "sum", "identity".
        """
        self.normalizer = get_normalizer(normalizer)

    def fuse(
        self,
        results_by_retriever: Dict[str, List[Tuple[str, float]]],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        if not results_by_retriever:
            return []

        all_docs = get_all_doc_ids(results_by_retriever)
        combsum_scores: Dict[str, float] = {doc_id: 0.0 for doc_id in all_docs}
        doc_counts: Dict[str, int] = {doc_id: 0 for doc_id in all_docs}

        for retriever_name, results in results_by_retriever.items():
            score_dict = results_to_score_dict(results)
            normalized = self.normalizer.normalize(score_dict)

            for doc_id in all_docs:
                score = normalized.get(doc_id, 0.0)
                if score > 0.0:
                    combsum_scores[doc_id] += score
                    doc_counts[doc_id] += 1

        combmnz_scores = {
            doc_id: doc_counts[doc_id] * combsum_scores[doc_id]
            for doc_id in all_docs
        }

        return sort_and_truncate(combmnz_scores, top_k)


class WeightedScoreFusion(FusionStrategy):
    """
    Weighted Score Fusion.

    Familia: Basada en scores.

    Fusión lineal ponderada entre dos recuperadores (típicamente sparse y dense).
    Permite controlar el balance entre ambos mediante el parámetro alpha.

    Fórmula:
        score(d) = alpha * normalized_sparse(d) + (1 - alpha) * normalized_dense(d)

    Parámetros:
        - alpha: peso del recuperador sparse [0, 1]
        - (1 - alpha): peso del recuperador dense

    Si un documento no aparece en uno de los recuperadores, se usa score 0.0
    para ese recuperador.

    Ventajas:
        - Control explícito del balance sparse/dense
        - Interpretable y simple
        - Permite optimizar alpha por tipo de query

    Limitaciones:
        - Solo funciona con exactamente 2 recuperadores
        - Requiere identificar cuál es sparse y cuál es dense

    Normalización requerida: Sí (default: minmax).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        sparse_key: str = "bm25",
        dense_key: str = "dense",
        normalizer: str = "minmax",
    ):
        """
        Args:
            alpha: Peso del recuperador sparse [0, 1]. Default: 0.5 (balance igual).
            sparse_key: Nombre del recuperador sparse en results_by_retriever.
            dense_key: Nombre del recuperador dense en results_by_retriever.
            normalizer: Normalizador a usar para ambos recuperadores.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha debe estar en [0, 1], recibido: {alpha}")

        self.alpha = alpha
        self.sparse_key = sparse_key
        self.dense_key = dense_key
        self.normalizer = get_normalizer(normalizer)

    def fuse(
        self,
        results_by_retriever: Dict[str, List[Tuple[str, float]]],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        if not results_by_retriever:
            return []

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

        sparse_results = results_by_retriever[self.sparse_key]
        dense_results = results_by_retriever[self.dense_key]

        sparse_scores = results_to_score_dict(sparse_results)
        dense_scores = results_to_score_dict(dense_results)

        sparse_normalized = self.normalizer.normalize(sparse_scores)
        dense_normalized = self.normalizer.normalize(dense_scores)

        all_docs = get_all_doc_ids(results_by_retriever)
        weighted_scores: Dict[str, float] = {}

        for doc_id in all_docs:
            sparse_score = sparse_normalized.get(doc_id, 0.0)
            dense_score = dense_normalized.get(doc_id, 0.0)
            weighted_scores[doc_id] = (
                self.alpha * sparse_score + (1.0 - self.alpha) * dense_score
            )

        return sort_and_truncate(weighted_scores, top_k)
