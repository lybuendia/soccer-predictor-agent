"""Anthropic Claude adapter implementing LLMProvider."""

from typing import Any
import anthropic


class ClaudeProvider:
    """Wraps anthropic.Anthropic to satisfy the LLMProvider protocol."""

    def __init__(self, client: anthropic.Anthropic, model: str = "claude-sonnet-4-6") -> None:
        """Initialise with an injected Anthropic client and target model name."""
        self._client = client
        self._model = model

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send messages and return the assistant content string."""
        system = kwargs.pop("system", None)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=kwargs.pop("max_tokens", 4096),
            messages=messages,
            **({"system": system} if system else {}),
            **kwargs,
        )
        return response.content[0].text

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        **kwargs: Any,
    ) -> dict:
        """Send messages with tool definitions; return a normalised response dict."""
        system = kwargs.pop("system", None)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=kwargs.pop("max_tokens", 4096),
            messages=messages,
            tools=tools,
            **({"system": system} if system else {}),
            **kwargs,
        )
        return {"content": [block.model_dump() for block in response.content], "stop_reason": response.stop_reason}

    def format_assistant_turn(self, response: dict) -> dict:
        """Wrap the content list as an assistant message for Anthropic message history."""
        return {"role": "assistant", "content": response.get("content", [])}

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        """Build an Anthropic tool_result user message."""
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": content}],
        }
