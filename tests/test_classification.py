"""Tests para clasificadores de consulta."""

import pytest
from unittest.mock import MagicMock

from src.adaptive.classification import (
    OracleQueryClassifier,
    RuleBasedQueryClassifier,
    LLMQueryClassifier,
    HybridQueryClassifier,
    QueryClassification,
    QUERY_TYPES,
)
from src.adaptive.query_signals import QuerySignalExtractor
from src.rag.generator.base import GenerationResult


class TestOracleQueryClassifier:
    def test_classify_by_id(self):
        type_map = {"q1": "referencia_exacta", "q2": "semantica", "q3": "multi_hop"}
        clf = OracleQueryClassifier(type_map)
        result = clf.classify_by_id("q1")
        assert result.query_type == "referencia_exacta"
        assert result.confidence == 1.0
        assert result.classifier == "oracle"

    def test_classify_raises(self):
        clf = OracleQueryClassifier({"q1": "semantica"})
        with pytest.raises(NotImplementedError):
            clf.classify("any query")

    def test_name(self):
        clf = OracleQueryClassifier({})
        assert clf.name == "oracle"


class TestRuleBasedQueryClassifier:
    def setup_method(self):
        self.clf = RuleBasedQueryClassifier()

    def test_referencia_exacta(self):
        result = self.clf.classify("Decreto-Ley 114 de 2025 sobre asociaciones")
        assert result.query_type == "referencia_exacta"
        assert result.confidence == 0.9

    def test_referencia_exacta_resolucion(self):
        result = self.clf.classify("Resolución 45 del año 2023 sobre comercio")
        assert result.query_type == "referencia_exacta"

    def test_temporal_historica(self):
        result = self.clf.classify(
            "en qué año se estableció el reglamento de comercio exterior"
        )
        assert result.query_type == "temporal_historica"
        assert result.confidence == 0.9

    def test_multi_hop(self):
        result = self.clf.classify(
            "qué norma crea el régimen especial y qué resolución establece el procedimiento"
        )
        assert result.query_type == "multi_hop"
        assert result.confidence == 0.9

    def test_multi_hop_deroga(self):
        result = self.clf.classify("qué decreto deroga la ley anterior de inversiones")
        assert result.query_type == "multi_hop"

    def test_ambigua_corta(self):
        result = self.clf.classify("licencia")
        assert result.query_type == "ambigua"
        assert result.confidence == 0.7

    def test_semantica_default(self):
        result = self.clf.classify("protección del medio ambiente en Cuba")
        assert result.query_type == "semantica"
        assert result.confidence == 0.55

    def test_all_outputs_valid_types(self):
        queries = [
            "decreto 5",
            "test",
            "qué norma modifica algo",
            "vigente la ley",
            "abc",
            "requisitos para constituir una empresa en Cuba según la legislación actual",
        ]
        for q in queries:
            result = self.clf.classify(q)
            assert result.query_type in QUERY_TYPES
            assert 0.0 <= result.confidence <= 1.0

    def test_classifier_name(self):
        result = self.clf.classify("test")
        assert result.classifier == "rule_based"


class TestLLMQueryClassifier:
    def _make_mock_generator(self, response_text: str) -> MagicMock:
        mock = MagicMock()
        mock.generate.return_value = GenerationResult(
            text=response_text, model="test-model"
        )
        return mock

    def test_valid_json_response(self):
        gen = self._make_mock_generator(
            '{"query_type": "semantica", "confidence": 0.85, "reason": "búsqueda conceptual"}'
        )
        clf = LLMQueryClassifier(gen)
        result = clf.classify("protección ambiental en Cuba")
        assert result.query_type == "semantica"
        assert result.confidence == 0.85
        assert result.classifier == "llm"

    def test_valid_json_with_surrounding_text(self):
        gen = self._make_mock_generator(
            'Aquí está mi respuesta: {"query_type": "multi_hop", "confidence": 0.9, "reason": "test"} fin'
        )
        clf = LLMQueryClassifier(gen)
        result = clf.classify("qué norma deroga la anterior")
        assert result.query_type == "multi_hop"

    def test_fallback_on_invalid_json(self):
        gen = self._make_mock_generator("Creo que es una consulta semántica sobre leyes")
        clf = LLMQueryClassifier(gen)
        result = clf.classify("protección ambiental")
        assert result.query_type in QUERY_TYPES
        assert "fallback" in result.classifier

    def test_fallback_on_invalid_type(self):
        gen = self._make_mock_generator(
            '{"query_type": "tipo_inventado", "confidence": 0.9, "reason": "test"}'
        )
        clf = LLMQueryClassifier(gen)
        result = clf.classify("algo")
        assert result.query_type in QUERY_TYPES
        assert "fallback" in result.classifier

    def test_fallback_on_exception(self):
        gen = MagicMock()
        gen.generate.side_effect = RuntimeError("API error")
        clf = LLMQueryClassifier(gen)
        result = clf.classify("test query")
        assert result.query_type in QUERY_TYPES
        assert "fallback" in result.classifier

    def test_confidence_clamped(self):
        gen = self._make_mock_generator(
            '{"query_type": "semantica", "confidence": 1.5, "reason": "test"}'
        )
        clf = LLMQueryClassifier(gen)
        result = clf.classify("test")
        assert result.confidence <= 1.0

    def test_retry_on_rate_limit(self):
        """Verifica que el clasificador reintenta en rate limits con backoff."""
        gen = MagicMock()
        # Primera llamada: rate limit error
        # Segunda llamada: éxito
        gen.generate.side_effect = [
            RuntimeError("API error occurred: Status 429. Body: rate limit exceeded"),
            GenerationResult(
                text='{"query_type": "semantica", "confidence": 0.8, "reason": "test"}',
                model="test",
            ),
        ]
        clf = LLMQueryClassifier(gen, max_retries=3, base_delay=0.01)  # delay mínimo para test
        result = clf.classify("test query")
        assert result.query_type == "semantica"
        assert gen.generate.call_count == 2  # 1 fallo + 1 éxito

    def test_exhausted_retries_fallback(self):
        """Verifica que después de todos los reintentos usa fallback."""
        gen = MagicMock()
        gen.generate.side_effect = RuntimeError("Status 429: rate_limited")
        clf = LLMQueryClassifier(gen, max_retries=2, base_delay=0.01)
        result = clf.classify("test query")
        assert result.query_type in QUERY_TYPES
        assert "exhausted" in result.classifier

    def test_name(self):
        gen = self._make_mock_generator("{}")
        clf = LLMQueryClassifier(gen)
        assert clf.name == "llm"


class TestHybridQueryClassifier:
    def test_uses_rules_when_high_confidence(self):
        rule_clf = RuleBasedQueryClassifier()
        llm_gen = MagicMock()
        llm_clf = LLMQueryClassifier(llm_gen)
        hybrid = HybridQueryClassifier(rule_clf, llm_clf)

        result = hybrid.classify("Decreto-Ley 114 de 2025 asociaciones")
        assert result.query_type == "referencia_exacta"
        assert "rules" in result.classifier
        llm_gen.generate.assert_not_called()

    def test_delegates_to_llm_on_low_confidence(self):
        rule_clf = RuleBasedQueryClassifier()
        llm_gen = MagicMock()
        llm_gen.generate.return_value = GenerationResult(
            text='{"query_type": "compleja_hibrida", "confidence": 0.85, "reason": "comparación"}',
            model="test",
        )
        llm_clf = LLMQueryClassifier(llm_gen)
        hybrid = HybridQueryClassifier(rule_clf, llm_clf)

        result = hybrid.classify("protección del medio ambiente en Cuba")
        assert result.query_type == "compleja_hibrida"
        assert "llm" in result.classifier
        llm_gen.generate.assert_called_once()

    def test_falls_back_to_rules_when_no_llm(self):
        rule_clf = RuleBasedQueryClassifier()
        hybrid = HybridQueryClassifier(rule_clf, llm_classifier=None)

        result = hybrid.classify("protección ambiental")
        assert result.query_type in QUERY_TYPES
        assert "no_llm" in result.classifier
