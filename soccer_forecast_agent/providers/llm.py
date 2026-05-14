"""LLMProvider protocol — the only interface agents use to call language models."""

from typing import Protocol, Any


class LLMProvider(Protocol):
    """Abstract interface for language model calls. Agents depend on this, never on a concrete SDK."""

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send a conversation and return the assistant reply as a string."""
        ...

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        **kwargs: Any,
    ) -> dict:
        """Send a conversation with tool definitions; return the raw response dict including any tool calls."""
        ...

    def format_assistant_turn(self, response: dict) -> dict:
        """Convert a raw chat_with_tools response into a message dict suitable for appending to history."""
        ...

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        """Build a tool-result message for appending to history after a tool was called."""
        ...
