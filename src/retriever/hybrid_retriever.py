from typing import Dict, List, Tuple

from .retriever import Retriever
from .fusion.base import FusionStrategy


class HybridRetriever(Retriever):
    """
    Recuperador híbrido que combina múltiples recuperadores mediante fusión.

    Flujo de operación:
    1. Ejecuta cada recuperador con candidate_k documentos
    2. Recolecta resultados en dict {nombre_recuperador: [(doc_id, score), ...]}
    3. Aplica la estrategia de fusión sobre todos los resultados
    4. Retorna el ranking fusionado truncado a top_k

    Ejemplo de uso:

        >>> from src.retriever import BM25Retriever, DenseRetriever
        >>> from src.retriever.fusion.strategies import ReciprocalRankFusion
        >>>
        >>> hybrid = HybridRetriever(
        ...     retrievers={
        ...         "bm25": BM25Retriever(),
        ...         "dense": DenseRetriever()
        ...     },
        ...     fusion_strategy=ReciprocalRankFusion(k=60),
        ...     candidate_k=50
        ... )
        >>>
        >>> results = hybrid.retrieve("licencia de maternidad", top_k=10)
    """

    def __init__(
        self,
        retrievers: Dict[str, Retriever],
        fusion_strategy: FusionStrategy,
        candidate_k: int = 50,
    ):
        """
        Args:
            retrievers: Dict mapeando nombre -> instancia de Retriever.
                       Los nombres deben coincidir con los esperados por
                       la estrategia de fusión (ej: "bm25", "dense").
            fusion_strategy: Estrategia de fusión a aplicar.
            candidate_k: Número de candidatos a recuperar de cada retriever
                        antes de fusionar. Debe ser >= top_k final.

        Raises:
            ValueError: Si retrievers está vacío o candidate_k <= 0.
        """
        if not retrievers:
            raise ValueError("Debe proporcionar al menos un recuperador")
        if candidate_k <= 0:
            raise ValueError(f"candidate_k debe ser > 0, recibido: {candidate_k}")

        self._retrievers = retrievers
        self._fusion_strategy = fusion_strategy
        self._candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Recupera documentos usando fusión de múltiples recuperadores.

        Args:
            query: Texto de la consulta.
            top_k: Número de resultados finales a retornar.

        Returns:
            Lista de (doc_id, score_fusionado) ordenada por score descendente.
        """
        results_by_retriever: Dict[str, List[Tuple[str, float]]] = {}

        for name, retriever in self._retrievers.items():
            results_by_retriever[name] = retriever.retrieve(query, self._candidate_k)

        return self._fusion_strategy.fuse(results_by_retriever, top_k)

    @property
    def name(self) -> str:
        """Retorna nombre descriptivo del recuperador híbrido."""
        strategy_name = self._fusion_strategy.__class__.__name__
        retriever_names = "+".join(self._retrievers.keys())
        return f"HybridRetriever({retriever_names}|{strategy_name})"
