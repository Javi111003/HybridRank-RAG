from __future__ import annotations

from typing import Any

from src.config import config
from .base import GeneratorProvider


def get_generator(name: str | None = None, **kwargs: Any) -> GeneratorProvider:
    """Factory para obtener un GeneratorProvider por nombre."""
    from .mistral_provider import MistralProvider
    from .litellm_provider import LiteLLMProvider

    name = name or config.GENERATOR_PROVIDER

    providers: dict[str, type[GeneratorProvider]] = {
        "mistral": MistralProvider,
        "litellm": LiteLLMProvider,
    }

    if name not in providers:
        raise ValueError(
            f"Proveedor desconocido: '{name}'. Disponibles: {list(providers.keys())}"
        )

    return providers[name](**kwargs)
