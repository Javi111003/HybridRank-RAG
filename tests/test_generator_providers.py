import pytest
from unittest.mock import MagicMock, patch

from src.rag.generator.base import GeneratorProvider, GenerationResult
from src.rag.generator.registry import get_generator


class FakeGeneratorProvider(GeneratorProvider):
    """Provider falso para tests."""

    def __init__(self, answer="respuesta de prueba", model="fake-model"):
        self._answer = answer
        self._model = model

    def generate(self, messages):
        return GenerationResult(
            text=self._answer,
            model=self._model,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            finish_reason="stop",
        )

    @property
    def name(self):
        return f"Fake({self._model})"


class TestGenerationResult:
    def test_dataclass_fields(self):
        result = GenerationResult(
            text="hola",
            model="test-model",
            usage={"total_tokens": 10},
            finish_reason="stop",
        )
        assert result.text == "hola"
        assert result.model == "test-model"
        assert result.usage["total_tokens"] == 10
        assert result.finish_reason == "stop"

    def test_optional_fields(self):
        result = GenerationResult(text="hola", model="test")
        assert result.usage is None
        assert result.finish_reason == ""


class TestFakeProvider:
    def test_generate(self):
        provider = FakeGeneratorProvider()
        messages = [{"role": "user", "content": "test"}]
        result = provider.generate(messages)

        assert result.text == "respuesta de prueba"
        assert result.model == "fake-model"
        assert result.usage["total_tokens"] == 150
        assert provider.name == "Fake(fake-model)"


class TestMistralProvider:
    @patch("src.rag.generator.mistral_provider.Mistral")
    def test_generate(self, MockMistral):
        mock_client = MagicMock()
        MockMistral.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "Respuesta sobre decreto"
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 200
        mock_usage.completion_tokens = 100
        mock_usage.total_tokens = 300

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "mistral-small-latest"
        mock_response.usage = mock_usage

        mock_client.chat.complete.return_value = mock_response

        from src.rag.generator.mistral_provider import MistralProvider

        provider = MistralProvider(api_key="test-key", model="mistral-small-latest")
        result = provider.generate([{"role": "user", "content": "test"}])

        assert result.text == "Respuesta sobre decreto"
        assert result.model == "mistral-small-latest"
        assert result.usage["total_tokens"] == 300


class TestLiteLLMProvider:
    def test_generate(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "Respuesta via LiteLLM"
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 150
        mock_usage.completion_tokens = 80
        mock_usage.total_tokens = 230

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "openrouter/mistralai/mistral-7b"
        mock_response.usage = mock_usage

        with patch("litellm.completion", return_value=mock_response):
            from src.rag.generator.litellm_provider import LiteLLMProvider

            provider = LiteLLMProvider(
                model="openrouter/mistralai/mistral-7b", api_key="test-key"
            )
            result = provider.generate([{"role": "user", "content": "test"}])

        assert result.text == "Respuesta via LiteLLM"
        assert result.usage["total_tokens"] == 230


class TestGetGenerator:
    @patch("src.rag.generator.mistral_provider.Mistral")
    def test_get_mistral(self, MockMistral):
        gen = get_generator("mistral", api_key="test-key")
        assert gen is not None

    def test_get_litellm(self):
        gen = get_generator("litellm", api_key="test-key")
        assert gen is not None

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Proveedor desconocido"):
            get_generator("unknown_provider")
