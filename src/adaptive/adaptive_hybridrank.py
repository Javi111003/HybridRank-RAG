"""Orquestador AdaptiveHybridRank: fusión adaptativa basada en tipo de consulta."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.retriever.retriever import Retriever
from .query_signals import QuerySignals, QuerySignalExtractor
from .classification import QueryClassifier, QueryClassification
from .policy import AdaptiveFusionPolicy, FusionConfig, GLOBAL_FALLBACK


class AdaptiveHybridRank:
    """
    Orquesta fusión adaptativa query-type-aware.

    Flujo:
    1. Recupera candidatos de BM25 y Dense
    2. Extrae señales de la query + resultados
    3. Clasifica el tipo de consulta
    4. Obtiene la mejor config de fusión para ese tipo
    5. Aplica la estrategia de fusión correspondiente
    """

    def __init__(
        self,
        retrievers: Dict[str, Retriever],
        classifier: QueryClassifier,
        policy: AdaptiveFusionPolicy,
        signal_extractor: Optional[QuerySignalExtractor] = None,
        global_fallback: Optional[FusionConfig] = None,
        candidate_k_default: int = 50,
        top_k: int = 10,
    ):
        self._retrievers = retrievers
        self._classifier = classifier
        self._policy = policy
        self._signal_extractor = signal_extractor or QuerySignalExtractor()
        self._fallback = global_fallback or GLOBAL_FALLBACK
        self._candidate_k = candidate_k_default
        self._top_k = top_k

    def retrieve(
        self,
        query: str,
        query_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Pipeline completo de retrieval adaptativo."""
        meta = self.retrieve_with_metadata(query, query_id, top_k)
        return meta["results"]

    def retrieve_with_metadata(
        self,
        query: str,
        query_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Pipeline adaptativo con metadata de clasificación.

        Returns:
            {
                "results": List[Tuple[str, float]],
                "query_type": str,
                "confidence": float,
                "classification": QueryClassification,
                "config": FusionConfig,
                "signals": QuerySignals,
                "fallback_used": bool,
            }
        """
        top_k = top_k or self._top_k

        results_by_retriever = self._retrieve_candidates(query)

        bm25_results = results_by_retriever.get("bm25", [])
        dense_results = results_by_retriever.get("dense", [])
        signals = self._signal_extractor.extract_with_retrieval(
            query, bm25_results, dense_results
        )

        classification = self._classify(query, query_id, signals)

        config = self._policy.select_config(
            classification.query_type,
            classification.confidence,
            self._policy.train_counts_by_type,
        )
        fallback_used = config == self._fallback

        strategy = config.build_strategy()
        fused = strategy.fuse(results_by_retriever, top_k=top_k)

        return {
            "results": fused,
            "query_type": classification.query_type,
            "confidence": classification.confidence,
            "classification": classification,
            "config": config,
            "signals": signals,
            "fallback_used": fallback_used,
        }

    def _retrieve_candidates(
        self, query: str
    ) -> Dict[str, List[Tuple[str, float]]]:
        results: Dict[str, List[Tuple[str, float]]] = {}
        for name, retriever in self._retrievers.items():
            results[name] = retriever.retrieve(query, self._candidate_k)
        return results

    def _classify(
        self,
        query: str,
        query_id: Optional[str],
        signals: QuerySignals,
    ) -> QueryClassification:
        from .classification import OracleQueryClassifier

        if isinstance(self._classifier, OracleQueryClassifier) and query_id:
            return self._classifier.classify_by_id(query_id)
        return self._classifier.classify(query, signals)
