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
