"""
Estrategias de fusión para HybridRank RAG.

Este módulo implementa múltiples estrategias para combinar rankings
de recuperadores sparse (BM25) y dense (embeddings):

Estrategias basadas en ranking:
    - ReciprocalRankFusion (RRF): Fusión basada en posiciones, robusta.
    - BordaFusion: Asigna puntos por posición relativa.

Estrategias basadas en scores:
    - CombSUMFusion: Suma de scores normalizados.
    - CombMNZFusion: CombSUM ponderado por número de recuperadores.
    - WeightedScoreFusion: Combinación lineal ponderada (2 retrievers).

Estrategias híbridas:
    - HybridRankFusion: Propuesta experimental que combina RRF + WeightedScore.

Uso típico:

    >>> from src.retriever import BM25Retriever, DenseRetriever
    >>> from src.retriever.fusion import get_fusion_strategy
    >>> from src.retriever.hybrid_retriever import HybridRetriever
    >>>
    >>> # Crear estrategia de fusión
    >>> strategy = get_fusion_strategy("hybridrank", alpha=0.7, beta=0.5)
    >>>
    >>> # Crear recuperador híbrido
    >>> hybrid = HybridRetriever(
    ...     retrievers={"bm25": BM25Retriever(), "dense": DenseRetriever()},
    ...     fusion_strategy=strategy,
    ...     candidate_k=50
    ... )
    >>>
    >>> # Recuperar
    >>> results = hybrid.retrieve("licencia de maternidad", top_k=10)
"""

from .base import FusionStrategy
from .strategies import (
    BordaFusion,
    CombMNZFusion,
    CombSUMFusion,
    ReciprocalRankFusion,
    WeightedScoreFusion,
)
from .hybrid_rank_fusion import HybridRankFusion
from .registry import get_fusion_strategy

__all__ = [
    "FusionStrategy",
    "ReciprocalRankFusion",
    "BordaFusion",
    "CombSUMFusion",
    "CombMNZFusion",
    "WeightedScoreFusion",
    "HybridRankFusion",
    "get_fusion_strategy",
]
