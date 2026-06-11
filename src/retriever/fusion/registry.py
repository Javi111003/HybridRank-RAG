from typing import Any

from .base import FusionStrategy
from .strategies import (
    BordaFusion,
    CombMNZFusion,
    CombSUMFusion,
    ReciprocalRankFusion,
    WeightedScoreFusion,
)
from .hybrid_rank_fusion import HybridRankFusion

STRATEGIES: dict[str, type[FusionStrategy]] = {
    "rrf": ReciprocalRankFusion,
    "borda": BordaFusion,
    "combsum": CombSUMFusion,
    "combmnz": CombMNZFusion,
    "weighted": WeightedScoreFusion,
    "hybridrank": HybridRankFusion,
}


def get_fusion_strategy(name: str, **kwargs: Any) -> FusionStrategy:
    strategy_cls = STRATEGIES.get(name)
    if strategy_cls is None:
        raise ValueError(
            f"Estrategia desconocida: '{name}'. "
            f"Disponibles: {list(STRATEGIES.keys())}"
        )

    return strategy_cls(**kwargs)
