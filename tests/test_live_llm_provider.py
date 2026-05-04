"""Opt-in live smoke tests for the real LLM provider adapters."""

from __future__ import annotations

import os

import pytest

from soccer_forecast_agent.config import Config
from soccer_forecast_agent.main import build_llm_provider
from soccer_forecast_agent.providers.llm import LLMProvider


def _live_test_config() -> Config:
    """Build a minimal Config for live LLM smoke tests from environment variables."""
    provider = os.environ.get("LLM_PROVIDER", "openai")
    return Config(
        llm_provider=provider,
        llm_model=os.environ.get("LLM_MODEL", "gpt-4o"),
        embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "huggingface"),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        football_data_api_key=os.environ.get("FOOTBALL_DATA_API_KEY", "test-football-key"),
        odds_api_key=os.environ.get("ODDS_API_KEY", "test-odds-key"),
        search_api_key=os.environ.get("SEARCH_API_KEY", "test-search-key"),
        smtp_host=os.environ.get("SMTP_HOST", "smtp.example.com"),
        smtp_port=int(os.environ.get("SMTP_PORT", "587")),
        smtp_user=os.environ.get("SMTP_USER", "user"),
        smtp_password=os.environ.get("SMTP_PASSWORD", "pass"),
        alert_email=os.environ.get("ALERT_EMAIL", "alerts@example.com"),
        db_path=os.environ.get("DB_PATH", "soccer_forecast.db"),
        chroma_path=os.environ.get("CHROMA_PATH", ".chroma"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )


def _require_live_provider() -> LLMProvider:
    """Return the configured live provider or skip if credentials are unavailable."""
    config = _live_test_config()
    if config.llm_provider == "openai" and not config.openai_api_key:
        pytest.skip("OPENAI_API_KEY is required for live OpenAI smoke tests.")
    if config.llm_provider == "claude" and not config.anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY is required for live Claude smoke tests.")
    return build_llm_provider(config)


@pytest.mark.live_llm
def test_live_llm_chat_returns_expected_smoke_response() -> None:
    """Verify the configured provider can answer a tiny deterministic prompt."""
    llm = _require_live_provider()

    response = llm.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "Reply with exactly the single word OK. "
                    "Do not add punctuation, explanation, or extra tokens."
                ),
            },
            {"role": "user", "content": "Return the required smoke-test reply."},
        ],
        temperature=0,
        max_tokens=16,
    )

    print(f"\nchat() raw response:\n{response}")

    assert response.strip() == "OK"


@pytest.mark.live_llm
def test_live_llm_tool_call_returns_expected_shape() -> None:
    """Verify the configured provider can emit a valid tool call payload."""
    llm = _require_live_provider()

    response = llm.chat_with_tools(
        messages=[
            {
                "role": "user",
                "content": (
                    "Use the search_news tool exactly once with query "
                    "'Arsenal Chelsea team news' and max_results 3."
                ),
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_news",
                    "description": "Search for recent soccer news.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer"},
                        },
                        "required": ["query", "max_results"],
                    },
                },
            }
        ],
        temperature=0,
        max_tokens=128,
    )

    print(f"\nchat_with_tools() raw response:\n{response}")

    tool_name, tool_args = _extract_tool_call(response)

    print(f"\nparsed tool call:\nname={tool_name}\nargs={tool_args}")

    assert tool_name == "search_news"
    assert tool_args["query"] == "Arsenal Chelsea team news"
    assert tool_args["max_results"] == 3


def _extract_tool_call(response: dict) -> tuple[str, dict]:
    """Normalise OpenAI and Anthropic tool-call responses into one assertion shape."""
    tool_calls = response.get("tool_calls") or []
    if tool_calls:
        function_call = tool_calls[0]["function"]
        arguments = function_call["arguments"]
        if isinstance(arguments, str):
            import json

            arguments = json.loads(arguments)
        return function_call["name"], arguments

    for block in response.get("content", []):
        if block.get("type") == "tool_use":
            return block["name"], block["input"]

    pytest.fail(f"Expected a tool call in live response, got: {response!r}")
