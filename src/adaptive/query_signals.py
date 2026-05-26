"""Extracción de señales observables de consultas para clasificación adaptativa."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class QuerySignals:
    """Señales observables extraídas de una consulta y sus resultados de retrieval."""

    query: str
    query_length_tokens: int
    has_legal_reference: bool
    has_norm_type: bool
    has_number: bool
    has_year: bool
    has_temporal_pattern: bool
    has_multihop_pattern: bool
    overlap_at_10: Optional[float] = None
    top1_bm25_score: Optional[float] = None
    top1_dense_score: Optional[float] = None
    bm25_dense_score_gap: Optional[float] = None
    metadata: dict = field(default_factory=dict)


class QuerySignalExtractor:
    """Extrae QuerySignals de texto de consulta y resultados de retrieval opcionales."""

    _NORM_TYPES = re.compile(
        r"(?i)\b(decreto[\s\-]*ley|decreto|ley|resolución|resolucion|"
        r"acuerdo|instrucción|instruccion|dictamen)\b"
    )
    _LEGAL_REF = re.compile(
        r"(?i)(decreto[\s\-]*ley|decreto|ley|resolución|resolucion|"
        r"acuerdo|instrucción|instruccion|dictamen|GOC)"
        r"[\s\-]*(?:No\.?\s*)?\d+"
    )
    _NUMBER = re.compile(r"(?:No\.?\s*|\b)\d{1,5}\b")
    _YEAR = re.compile(r"\b(19|20)\d{2}\b")
    _TEMPORAL = re.compile(
        r"(?i)(en qué año|cuándo|cuando|fecha|vigente|vigencia|"
        r"aprobó|aprobo|surgió|surgio|se estableció|se establecio|origen)"
    )
    _MULTIHOP = re.compile(
        r"(?i)(qué norma.*y qué|que norma.*y que|"
        r"qué norma.*establece|que norma.*establece|"
        r"qué resolución.*deroga|que resolucion.*deroga|"
        r"modifica|deroga|complementa|relación entre|relacion entre|"
        r"cuál.*y cuál|cual.*y cual|"
        r"qué norma crea.*y qué|que norma crea.*y que)"
    )

    def extract_static(self, query: str) -> QuerySignals:
        """Extrae señales basadas únicamente en el texto de la consulta."""
        tokens = query.split()
        return QuerySignals(
            query=query,
            query_length_tokens=len(tokens),
            has_legal_reference=bool(self._LEGAL_REF.search(query)),
            has_norm_type=bool(self._NORM_TYPES.search(query)),
            has_number=bool(self._NUMBER.search(query)),
            has_year=bool(self._YEAR.search(query)),
            has_temporal_pattern=bool(self._TEMPORAL.search(query)),
            has_multihop_pattern=bool(self._MULTIHOP.search(query)),
        )

    def extract_with_retrieval(
        self,
        query: str,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        k: int = 10,
    ) -> QuerySignals:
        """Extrae señales estáticas + señales de retrieval."""
        signals = self.extract_static(query)

        top_k_bm25 = bm25_results[:k]
        top_k_dense = dense_results[:k]

        bm25_ids = {doc_id for doc_id, _ in top_k_bm25}
        dense_ids = {doc_id for doc_id, _ in top_k_dense}

        overlap = len(bm25_ids & dense_ids) / k if k > 0 else 0.0
        signals.overlap_at_10 = overlap

        signals.top1_bm25_score = bm25_results[0][1] if bm25_results else None
        signals.top1_dense_score = dense_results[0][1] if dense_results else None

        if signals.top1_bm25_score is not None and signals.top1_dense_score is not None:
            signals.bm25_dense_score_gap = (
                signals.top1_bm25_score - signals.top1_dense_score
            )

        return signals
