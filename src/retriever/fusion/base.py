from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class FusionStrategy(ABC):
    """
    Interfaz abstracta para estrategias de fusión de rankings.

    Las estrategias combinan resultados de múltiples recuperadores
    en un ranking unificado. Pueden operar sobre scores, posiciones,
    o una combinación de ambos.
    """

    @abstractmethod
    def fuse(
        self,
        results_by_retriever: Dict[str, List[Tuple[str, float]]],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Fusiona resultados de múltiples recuperadores.

        Args:
            results_by_retriever: Dict nombre_recuperador -> [(doc_id, score), ...]
                                  Cada lista ordenada por score descendente.
            top_k: Número máximo de resultados a retornar.

        Returns:
            Lista de (doc_id, score_final) ordenada descendente, truncada a top_k.
        """
        pass
