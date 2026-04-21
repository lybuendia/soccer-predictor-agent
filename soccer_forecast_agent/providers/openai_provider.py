"""OpenAI adapter implementing LLMProvider."""

from typing import Any
import openai


class OpenAIProvider:
    """Wraps openai.OpenAI to satisfy the LLMProvider protocol."""

    def __init__(self, client: openai.OpenAI, model: str = "gpt-4o") -> None:
        """Initialise with an injected OpenAI client and target model name."""
        self._client = client
        self._model = model

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send messages and return the assistant content string."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        **kwargs: Any,
    ) -> dict:
        """Send messages with tool definitions; return the raw response as a dict."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            **kwargs,
        )
        return response.choices[0].message.model_dump()
