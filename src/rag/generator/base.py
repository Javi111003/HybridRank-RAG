from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GenerationResult:
    """Resultado de generacion desde un LLM."""

    text: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: str = ""


class GeneratorProvider(ABC):
    """Interfaz abstracta para proveedores de generacion de texto."""

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> GenerationResult:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
