from __future__ import annotations

from typing import TypeAlias

from .templates import SYSTEM_PROMPT_ES, USER_PROMPT_TEMPLATE

ChatMessage: TypeAlias = dict[str, str]


class PromptBuilder:
    """Builds OpenAI-compatible chat messages."""

    def __init__(self, system_prompt: str | None = None):
        self._system_prompt = system_prompt or SYSTEM_PROMPT_ES

    def build(self, query: str, context: str) -> list[ChatMessage]:
        return [
            self._message("system", self._system_prompt),
            self._message("user", self._build_user_prompt(query, context)),
        ]

    @staticmethod
    def _message(role: str, content: str) -> ChatMessage:
        return {"role": role, "content": content}

    @staticmethod
    def _build_user_prompt(query: str, context: str) -> str:
        return USER_PROMPT_TEMPLATE.format(context=context, query=query)
