from typing import Any, Dict

from .base import FusionStrategy
from .strategies import (
    BordaFusion,
    CombMNZFusion,
    CombSUMFusion,
    ReciprocalRankFusion,
    WeightedScoreFusion,
)
from .hybrid_rank_fusion import HybridRankFusion


def get_fusion_strategy(name: str, **kwargs: Any) -> FusionStrategy:
    """
    Factory de estrategias de fusión.

    Crea instancias de estrategias de fusión por nombre, pasando
    parámetros de configuración opcionales.

    Args:
        name: Nombre de la estrategia:
            - "rrf": ReciprocalRankFusion
            - "borda": BordaFusion
            - "combsum": CombSUMFusion
            - "combmnz": CombMNZFusion
            - "weighted": WeightedScoreFusion
            - "hybridrank": HybridRankFusion
        **kwargs: Parámetros específicos de cada estrategia.

    Returns:
        Instancia configurada de FusionStrategy.

    Raises:
        ValueError: Si el nombre de estrategia no es reconocido.

    Ejemplos:

        >>> # RRF con k=100
        >>> strategy = get_fusion_strategy("rrf", k=100)
        >>>
        >>> # Weighted fusion con alpha=0.7
        >>> strategy = get_fusion_strategy("weighted", alpha=0.7)
        >>>
        >>> # HybridRank con parámetros personalizados
        >>> strategy = get_fusion_strategy(
        ...     "hybridrank",
        ...     alpha=0.7,
        ...     beta=0.5,
        ...     k=60
        ... )
    """
    strategies: Dict[str, type] = {
        "rrf": ReciprocalRankFusion,
        "borda": BordaFusion,
        "combsum": CombSUMFusion,
        "combmnz": CombMNZFusion,
        "weighted": WeightedScoreFusion,
        "hybridrank": HybridRankFusion,
    }

    if name not in strategies:
        raise ValueError(
            f"Estrategia desconocida: '{name}'. "
            f"Disponibles: {list(strategies.keys())}"
        )

    return strategies[name](**kwargs)
