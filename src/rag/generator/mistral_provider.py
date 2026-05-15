from __future__ import annotations

import logging
from typing import Dict, List

from mistralai.client import Mistral

from src.config import config
from .base import GeneratorProvider, GenerationResult

logger = logging.getLogger(__name__)


class MistralProvider(GeneratorProvider):
    """Proveedor de generacion usando Mistral AI oficial."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self._api_key = api_key or config.MISTRAL_API_KEY
        if not self._api_key:
            raise ValueError(
                "MISTRAL_API_KEY no configurada. "
                "Configurala en .env o pasala como parametro."
            )

        self._client = Mistral(api_key=self._api_key)
        self._model = model or config.GENERATOR_MODEL
        self._temperature = (
            temperature if temperature is not None else config.GENERATOR_TEMPERATURE
        )
        self._max_tokens = max_tokens or config.GENERATOR_MAX_TOKENS
        logger.info("MistralProvider inicializado: modelo=%s", self._model)

    def generate(self, messages: List[Dict[str, str]]) -> GenerationResult:
        logger.info(
            "Generando con %s (temp=%.2f, max_tokens=%d)",
            self._model,
            self._temperature,
            self._max_tokens,
        )

        response = self._client.chat.complete(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        choice = response.choices[0]
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        logger.info("Generacion completa: %d tokens totales", usage["total_tokens"])

        return GenerationResult(
            text=choice.message.content,
            model=response.model,
            usage=usage,
            finish_reason=choice.finish_reason or "",
        )

    @property
    def name(self) -> str:
        return f"Mistral({self._model})"
