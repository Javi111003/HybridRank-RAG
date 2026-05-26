"""Clasificadores de consulta para fusión adaptativa."""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from .query_signals import QuerySignals, QuerySignalExtractor
from src.rag.generator.base import GeneratorProvider

logger = logging.getLogger(__name__)

QUERY_TYPES = [
    "referencia_exacta",
    "semantica",
    "compleja_hibrida",
    "ambigua",
    "multi_hop",
    "temporal_historica",
]


@dataclass
class QueryClassification:
    """Resultado de la clasificación de una consulta."""

    query_type: str
    confidence: float
    reason: str
    classifier: str
    signals: dict


class QueryClassifier(ABC):
    """Interfaz abstracta para clasificadores de tipo de consulta."""

    @abstractmethod
    def classify(
        self, query: str, signals: Optional[QuerySignals] = None
    ) -> QueryClassification:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class OracleQueryClassifier(QueryClassifier):
    """
    Clasificador oracle que usa el query_type real del dataset.
    Solo para estimación de upper bound en experimentos.
    """

    def __init__(self, query_type_map: Dict[str, str]):
        self._type_map = query_type_map

    def classify(
        self, query: str, signals: Optional[QuerySignals] = None
    ) -> QueryClassification:
        raise NotImplementedError(
            "OracleQueryClassifier requiere query_id. Usar classify_by_id()."
        )

    def classify_by_id(self, query_id: str) -> QueryClassification:
        query_type = self._type_map[query_id]
        return QueryClassification(
            query_type=query_type,
            confidence=1.0,
            reason="ground_truth",
            classifier=self.name,
            signals={},
        )

    @property
    def name(self) -> str:
        return "oracle"


class RuleBasedQueryClassifier(QueryClassifier):
    """Clasificador basado en reglas usando señales extraídas de la consulta."""

    def __init__(self, signal_extractor: Optional[QuerySignalExtractor] = None):
        self._extractor = signal_extractor or QuerySignalExtractor()

    def classify(
        self, query: str, signals: Optional[QuerySignals] = None
    ) -> QueryClassification:
        if signals is None:
            signals = self._extractor.extract_static(query)

        query_type, confidence, reason = self._apply_rules(signals)
        return QueryClassification(
            query_type=query_type,
            confidence=confidence,
            reason=reason,
            classifier=self.name,
            signals={
                "has_legal_reference": signals.has_legal_reference,
                "has_year": signals.has_year,
                "has_number": signals.has_number,
                "has_temporal_pattern": signals.has_temporal_pattern,
                "has_multihop_pattern": signals.has_multihop_pattern,
                "query_length_tokens": signals.query_length_tokens,
            },
        )

    def _apply_rules(
        self, signals: QuerySignals
    ) -> tuple[str, float, str]:
        if signals.has_legal_reference and (signals.has_number or signals.has_year):
            return "referencia_exacta", 0.9, "referencia legal con número o año"

        if signals.has_temporal_pattern:
            return "temporal_historica", 0.9, "patrón temporal detectado"

        if signals.has_multihop_pattern:
            return "multi_hop", 0.9, "patrón multi-hop detectado"

        if signals.query_length_tokens <= 3 and not signals.has_legal_reference:
            return "ambigua", 0.7, "consulta corta sin referencia legal"

        if signals.has_norm_type and signals.query_length_tokens > 12:
            return "compleja_hibrida", 0.7, "tipo de norma con consulta extensa"

        return "semantica", 0.55, "fallback semántico"

    @property
    def name(self) -> str:
        return "rule_based"


_LLM_SYSTEM_PROMPT = """\
Eres un clasificador de consultas para un sistema RAG jurídico sobre legislación cubana.

Tu tarea es clasificar la consulta del usuario en exactamente una de estas categorías:

1. referencia_exacta:
Consultas que mencionan explícitamente tipo de norma, número, año, organismo, código GOC o nombre exacto de una norma.

2. semantica:
Consultas en lenguaje natural que buscan una idea jurídica sin mencionar una norma exacta.

3. compleja_hibrida:
Consultas que combinan términos jurídicos específicos con una necesidad explicativa amplia, comparación, requisitos, documentos o procedimiento.

4. ambigua:
Consultas cortas o generales que pueden referirse a varias normas, instituciones o sentidos jurídicos.

5. multi_hop:
Consultas que requieren conectar dos o más normas, relaciones normativas, derogaciones, modificaciones o procedimientos vinculados.

6. temporal_historica:
Consultas que preguntan por el año, origen, vigencia, fecha de aprobación o evolución temporal de una norma o institución.

Devuelve SOLO JSON válido con esta estructura:

{"query_type": "...", "confidence": 0.0, "reason": "..."}

Ejemplos:
- "Decreto-Ley 114 de 2025 asociación entre entidades empresariales estatales y no estatales" -> {"query_type": "referencia_exacta", "confidence": 0.95, "reason": "mención explícita de Decreto-Ley con número y año"}
- "qué requisitos debe cumplir una sociedad de responsabilidad limitada mixta para constituirse en Cuba" -> {"query_type": "semantica", "confidence": 0.9, "reason": "búsqueda conceptual sin referencia a norma específica"}
- "diferencias entre sociedad de responsabilidad limitada mixta y contrato de asociación económica" -> {"query_type": "compleja_hibrida", "confidence": 0.85, "reason": "comparación entre conceptos jurídicos específicos"}
- "licencia" -> {"query_type": "ambigua", "confidence": 0.9, "reason": "término genérico sin contexto"}
- "qué norma crea el régimen especial y qué resolución establece su procedimiento" -> {"query_type": "multi_hop", "confidence": 0.9, "reason": "requiere conectar dos normas diferentes"}
- "en qué año se estableció el reglamento para las representaciones comerciales extranjeras en Cuba" -> {"query_type": "temporal_historica", "confidence": 0.9, "reason": "pregunta por fecha de establecimiento"}"""


class LLMQueryClassifier(QueryClassifier):
    """
    Clasificador que usa un LLM para determinar el tipo de consulta.
    El LLM solo clasifica la consulta, NO selecciona documentos.
    """

    def __init__(self, generator: GeneratorProvider, max_retries: int = 5, base_delay: float = 1.0):
        self._generator = generator
        self._fallback = RuleBasedQueryClassifier()
        self._max_retries = max_retries
        self._base_delay = base_delay

    def classify(
        self, query: str, signals: Optional[QuerySignals] = None
    ) -> QueryClassification:
        last_error = None

        for attempt in range(self._max_retries):
            try:
                messages: List[Dict[str, str]] = [
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": f'Consulta:\n"{query}"'},
                ]
                result = self._generator.generate(messages)
                return self._parse_response(result.text, query)
            except Exception as e:
                last_error = e
                error_msg = str(e)
                # Detectar rate limit errors
                is_rate_limit = (
                    "429" in error_msg
                    or "rate limit" in error_msg.lower()
                    or "rate_limited" in error_msg.lower()
                )

                # Si es rate limit y no es el último intento, reintentar con backoff
                if is_rate_limit and attempt < self._max_retries - 1:
                    delay = self._base_delay * (2 ** attempt)
                    logger.warning(
                        "Rate limit hit (attempt %d/%d). Retrying in %.1fs...",
                        attempt + 1,
                        self._max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                # Para otros errores (no rate limit), no reintentar
                if not is_rate_limit:
                    logger.warning("LLM classification failed: %s. Using rule-based fallback.", e)
                    fallback = self._fallback.classify(query, signals)
                    fallback.classifier = "llm_fallback"
                    return fallback

        # Si llegamos aquí, todos los reintentos se agotaron (solo para rate limits)
        logger.error("All %d rate limit retries exhausted. Using fallback.", self._max_retries)
        fallback = self._fallback.classify(query, signals)
        fallback.classifier = "llm_exhausted_fallback"
        return fallback

    def _parse_response(self, text: str, query: str) -> QueryClassification:
        parsed = self._try_parse_json(text)
        if parsed is None:
            logger.warning("Failed to parse LLM JSON response: %s", text[:200])
            fallback = self._fallback.classify(query)
            fallback.classifier = "llm_parse_fallback"
            return fallback

        query_type = parsed.get("query_type", "")
        if query_type not in QUERY_TYPES:
            logger.warning("LLM returned invalid type: %s", query_type)
            fallback = self._fallback.classify(query)
            fallback.classifier = "llm_invalid_type_fallback"
            return fallback

        confidence = float(parsed.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))
        reason = parsed.get("reason", "")

        return QueryClassification(
            query_type=query_type,
            confidence=confidence,
            reason=reason,
            classifier=self.name,
            signals={},
        )

    @staticmethod
    def _try_parse_json(text: str) -> Optional[dict]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[^{}]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    @property
    def name(self) -> str:
        return "llm"


class HybridQueryClassifier(QueryClassifier):
    """
    Combina reglas + LLM: usa rules cuando tienen alta confianza,
    delega al LLM cuando las reglas son ambiguas.
    """

    def __init__(
        self,
        rule_classifier: Optional[RuleBasedQueryClassifier] = None,
        llm_classifier: Optional[LLMQueryClassifier] = None,
        rule_confidence_threshold: float = 0.85,
        llm_confidence_threshold: float = 0.65,
    ):
        self._rules = rule_classifier or RuleBasedQueryClassifier()
        self._llm = llm_classifier
        self._rule_threshold = rule_confidence_threshold
        self._llm_threshold = llm_confidence_threshold

    def classify(
        self, query: str, signals: Optional[QuerySignals] = None
    ) -> QueryClassification:
        rule_result = self._rules.classify(query, signals)

        if rule_result.confidence >= self._rule_threshold:
            rule_result.classifier = "hybrid_rules"
            return rule_result

        if self._llm is None:
            rule_result.classifier = "hybrid_rules_no_llm"
            return rule_result

        llm_result = self._llm.classify(query, signals)

        if llm_result.confidence >= self._llm_threshold and "fallback" not in llm_result.classifier:
            llm_result.classifier = "hybrid_llm"
            return llm_result

        rule_result.classifier = "hybrid_rules_llm_low_conf"
        return rule_result

    @property
    def name(self) -> str:
        return "hybrid"
