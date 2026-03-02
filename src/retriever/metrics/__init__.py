# Metrics module for evaluating retrieval systems
"""
Este módulo contiene implementaciones de métricas de evaluación para
sistemas de recuperación de información (Information Retrieval).

Métricas disponibles:
- RecallAtK: Mide la proporción de documentos relevantes recuperados en top-k
- PrecisionAtK: Mide la proporción de documentos relevantes en top-k (limpieza)
- F1AtK: Balance harmónico entre Precision@k y Recall@k
- MRR: Mean Reciprocal Rank - posición del primer documento relevante
- MAP: Mean Average Precision - promedio de precisiones en posiciones relevantes
- NDCG: Normalized Discounted Cumulative Gain - calidad del ranking con descuento posicional
"""

from .base import Metric
from .recall import RecallAtK
from .precision import PrecisionAtK
from .f1_score import F1AtK
from .mrr import MRR
from .map_metric import MAP
from .ndcg import NDCG

__all__ = [
    "Metric",
    "RecallAtK",
    "PrecisionAtK",
    "F1AtK",
    "MRR",
    "MAP",
    "NDCG",
]
