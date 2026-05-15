from __future__ import annotations

from typing import Dict, List

from .templates import SYSTEM_PROMPT_ES, USER_PROMPT_TEMPLATE


class PromptBuilder:
    """Construye mensajes para APIs de chat (formato OpenAI-compatible)."""

    def __init__(self, system_prompt: str | None = None):
        self._system_prompt = system_prompt or SYSTEM_PROMPT_ES

    def build(self, query: str, context: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(context=context, query=query),
            },
        ]
