from __future__ import annotations

import logging
from typing import Dict, List

from src.config import config
from .base import GeneratorProvider, GenerationResult

logger = logging.getLogger(__name__)


class LiteLLMProvider(GeneratorProvider):
    """Proveedor de generacion usando LiteLLM (soporta OpenRouter y otros backends)."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self._model = model or "openrouter/mistralai/mistral-small-3.1-24b-instruct"
        self._api_key = api_key or config.OPENROUTER_API_KEY
        self._api_base = api_base
        self._temperature = (
            temperature if temperature is not None else config.GENERATOR_TEMPERATURE
        )
        self._max_tokens = max_tokens or config.GENERATOR_MAX_TOKENS
        logger.info("LiteLLMProvider inicializado: modelo=%s", self._model)

    def generate(self, messages: List[Dict[str, str]]) -> GenerationResult:
        import litellm

        logger.info(
            "Generando con %s via LiteLLM (temp=%.2f, max_tokens=%d)",
            self._model,
            self._temperature,
            self._max_tokens,
        )

        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base

        response = litellm.completion(**kwargs)

        choice = response.choices[0]
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            logger.info("Generacion completa: %d tokens totales", usage["total_tokens"])

        return GenerationResult(
            text=choice.message.content,
            model=response.model or self._model,
            usage=usage,
            finish_reason=choice.finish_reason or "",
        )

    @property
    def name(self) -> str:
        return f"LiteLLM({self._model})"
