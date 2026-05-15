import pytest

from src.rag.prompt.prompt_builder import PromptBuilder
from src.rag.prompt.templates import SYSTEM_PROMPT_ES


class TestPromptBuilder:
    def test_build_returns_messages_list(self):
        builder = PromptBuilder()
        messages = builder.build("pregunta de prueba", "contexto de prueba")

        assert isinstance(messages, list)
        assert len(messages) == 2

    def test_build_system_message(self):
        builder = PromptBuilder()
        messages = builder.build("pregunta", "contexto")

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT_ES

    def test_build_user_message_contains_query_and_context(self):
        builder = PromptBuilder()
        messages = builder.build("que dice el decreto 114", "texto del decreto")

        user_content = messages[1]["content"]
        assert messages[1]["role"] == "user"
        assert "que dice el decreto 114" in user_content
        assert "texto del decreto" in user_content

    def test_custom_system_prompt(self):
        custom = "Eres un bot generico."
        builder = PromptBuilder(system_prompt=custom)
        messages = builder.build("pregunta", "contexto")

        assert messages[0]["content"] == custom

    def test_user_message_mentions_fuentes(self):
        builder = PromptBuilder()
        messages = builder.build("pregunta", "contexto")
        assert "[Fuente N]" in messages[1]["content"]
