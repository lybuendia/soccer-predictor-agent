"""Opt-in live smoke test for NewsContextAgent with controlled retrieval and tool inputs."""

from __future__ import annotations

import os

import pytest

from soccer_forecast_agent.agents.news_context import NewsContextAgent, ToolDispatcher
from soccer_forecast_agent.config import Config
from soccer_forecast_agent.main import build_llm_provider
from soccer_forecast_agent.models.evidence import ArticleChunk, EvidenceItem
from soccer_forecast_agent.providers.llm import LLMProvider


class FakeVectorRepository:
    """Provide controlled seeded context without requiring a real vector store."""

    def __init__(self, chunks: list[ArticleChunk]) -> None:
        self._chunks = chunks
        self.calls: list[dict] = []

    def search(self, query: str, teams: list[str], top_k: int = 5) -> list[ArticleChunk]:
        self.calls.append({"query": query, "teams": teams, "top_k": top_k})
        return self._chunks


class FakeEvidenceRepository:
    """Capture saved evidence items for assertions and inspection."""

    def __init__(self) -> None:
        self.saved: list[EvidenceItem] = []

    def save_evidence(self, item: EvidenceItem) -> None:
        self.saved.append(item)


class FakeSourceReliabilityRepository:
    """Return deterministic reliability scores for known domains."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def get_score(self, source_domain: str) -> float:
        return self._scores.get(source_domain, 0.5)


class FakeMCPClient:
    """Return a fixed live-search payload while recording tool calls."""

    def __init__(self, result: object) -> None:
        self._result = result
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, tool_name: str, arguments: dict) -> object:
        self.calls.append((tool_name, arguments))
        return self._result


def _live_test_config() -> Config:
    """Build a minimal config from environment variables for opt-in live tests."""
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


def _base_state(sample_match, sample_baseline) -> dict:
    """Return the minimal graph state required by NewsContextAgent."""
    return {
        "competition": "PL",
        "days_ahead": 7,
        "matches": [sample_match],
        "odds_map": {},
        "baseline_forecasts": {sample_match.match_id: sample_baseline},
        "current_match_id": sample_match.match_id,
        "evidence_items": [],
        "forecast": None,
        "alert_sent": False,
        "errors": [],
    }


@pytest.mark.live_llm
def test_live_news_context_agent_run_produces_structured_evidence(sample_match, sample_baseline) -> None:
    """Run NewsContextAgent with a real LLM and controlled tool data, then validate the output shape."""
    llm = _require_live_provider()
    vector_repo = FakeVectorRepository(
        [
            ArticleChunk(
                chunk_id="chunk-1",
                content=(
                    "Arsenal training report says the first-choice striker returned to full training and is in line "
                    "to start against Chelsea."
                ),
                source="bbc.co.uk",
                url="https://bbc.co.uk/sport/football/example-training-report",
                published_at="2026-04-24T10:00:00+00:00",
                teams=["Arsenal"],
            )
        ]
    )
    evidence_repo = FakeEvidenceRepository()
    mcp_client = FakeMCPClient(
        result=[
            {
                "url": "https://skysports.com/example-arsenal-chelsea-preview",
                "title": "Arsenal vs Chelsea team news and preview",
                "snippet": (
                    "Chelsea rotated in midweek, while Arsenal expect their leading striker to be available and may "
                    "name an unchanged attacking unit."
                ),
            }
        ]
    )
    agent = NewsContextAgent(
        llm=llm,
        tool_dispatcher=ToolDispatcher(mcp_client),
        vector_repo=vector_repo,
        evidence_repo=evidence_repo,
        source_reliability_repo=FakeSourceReliabilityRepository({"bbc.co.uk": 0.85, "skysports.com": 0.85}),
        max_steps=2,
        min_evidence_count=2,
        min_avg_reliability=0.5,
    )

    result = agent.run(_base_state(sample_match, sample_baseline))

    print(f"\nvector search calls:\n{vector_repo.calls}")
    print(f"\nMCP tool calls:\n{mcp_client.calls}")
    print("\nSaved evidence:")
    for item in evidence_repo.saved:
        print(
            f"- source={item.source} direction={item.direction} market={item.applies_to_market} "
            f"reliability={item.reliability_score} summary={item.summary}"
        )
    print(f"\nAgent errors:\n{result['errors']}")

    assert result["errors"] == []
    assert result["evidence_items"]
    assert evidence_repo.saved
    assert mcp_client.calls
    assert len(result["evidence_items"]) == len(evidence_repo.saved)
    assert len(result["evidence_items"]) >= 2
    assert all(item.summary for item in result["evidence_items"])
    assert all(item.direction in NewsContextAgent.VALID_DIRECTIONS for item in result["evidence_items"])
    assert all(item.applies_to_market in NewsContextAgent.VALID_MARKETS for item in result["evidence_items"])
