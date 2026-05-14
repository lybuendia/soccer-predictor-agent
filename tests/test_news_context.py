from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from soccer_forecast_agent.agents.news_context import NewsContextAgent, ToolDispatcher
from soccer_forecast_agent.models.evidence import ArticleChunk, EvidenceItem, InterpretedEvidence


@dataclass
class FakeMCPClient:
    result: object

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, tool_name: str, arguments: dict) -> object:
        self.calls.append((tool_name, arguments))
        return self.result


class FakeLLM:
    def __init__(self, tool_responses: list[dict], chat_responses: list[str]) -> None:
        self._tool_responses = list(tool_responses)
        self._chat_responses = list(chat_responses)
        self.tool_calls: list[dict] = []
        self.chat_calls: list[dict] = []

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return self._chat_responses.pop(0)

    def chat_with_tools(self, messages: list[dict[str, str]], tools: list[dict], **kwargs) -> dict:
        self.tool_calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        return self._tool_responses.pop(0)


class FakeVectorRepository:
    def __init__(self, chunks: list[ArticleChunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict] = []

    def search(self, query: str, teams: list[str], top_k: int = 5) -> list[ArticleChunk]:
        self.calls.append({"query": query, "teams": teams, "top_k": top_k})
        return self.chunks


class FakeEvidenceRepository:
    def __init__(self) -> None:
        self.saved: list[EvidenceItem] = []

    def save_evidence(self, item: EvidenceItem) -> None:
        self.saved.append(item)


class FailingEvidenceRepository(FakeEvidenceRepository):
    def save_evidence(self, item: EvidenceItem) -> None:
        raise RuntimeError("database unavailable")


class FakeSourceReliabilityRepository:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def get_score(self, source_domain: str) -> float:
        return self.scores.get(source_domain, 0.5)


def _base_state(sample_match, sample_baseline) -> dict:
    return {
        "competition": "PL",
        "days_ahead": 7,
        "matches": [sample_match],
        "odds_map": {},
        "baseline_forecasts": {sample_match.match_id: sample_baseline},
        "current_match_id": sample_match.match_id,
        "evidence_items": [],
        "interpreted_evidence": [],
        "forecast": None,
        "alert_sent": False,
        "errors": [],
    }


def test_tool_dispatcher_routes_calls_and_serialises_results() -> None:
    client = FakeMCPClient(result=[{"title": "Team news", "url": "https://bbc.co.uk/story"}])
    dispatcher = ToolDispatcher(client)

    result = dispatcher.dispatch("search_news", {"query": "Arsenal Chelsea team news", "max_results": 3})

    assert client.calls == [("search_news", {"query": "Arsenal Chelsea team news", "max_results": 3})]
    assert "Team news" in result
    assert "bbc.co.uk" in result


def test_seed_context_queries_vector_repo_for_both_teams(sample_match) -> None:
    chunk = ArticleChunk(
        chunk_id="chunk-1",
        content="Arsenal striker returned to training this week.",
        source="bbc.co.uk",
        url="https://bbc.co.uk/story",
        published_at="2026-04-24T10:00:00+00:00",
        teams=["Arsenal"],
    )
    agent = NewsContextAgent(
        llm=FakeLLM(tool_responses=[], chat_responses=[]),
        tool_dispatcher=ToolDispatcher(FakeMCPClient(result=[])),
        vector_repo=FakeVectorRepository([chunk]),
        evidence_repo=FakeEvidenceRepository(),
        source_reliability_repo=FakeSourceReliabilityRepository({}),
    )

    seeded = agent._seed_context(sample_match.home_team, sample_match.away_team)

    assert len(seeded) == 1
    assert "bbc.co.uk" in seeded[0]
    assert "Arsenal striker returned to training" in seeded[0]
    assert agent._vector.calls == [
        {
            "query": "Arsenal vs Chelsea injuries suspensions form lineup team news",
            "teams": ["Arsenal", "Chelsea"],
            "top_k": 5,
        }
    ]


def test_should_stop_requires_enough_high_reliability_evidence(sample_match) -> None:
    agent = NewsContextAgent(
        llm=FakeLLM(tool_responses=[], chat_responses=[]),
        tool_dispatcher=ToolDispatcher(FakeMCPClient(result=[])),
        vector_repo=FakeVectorRepository([]),
        evidence_repo=FakeEvidenceRepository(),
        source_reliability_repo=FakeSourceReliabilityRepository({}),
        max_steps=4,
        min_evidence_count=2,
        min_avg_reliability=0.7,
    )
    evidence = [
        EvidenceItem(
            evidence_id="e1",
            forecast_id=sample_match.match_id,
            source="bbc.co.uk",
            url="https://bbc.co.uk/1",
            timestamp=datetime.now(timezone.utc),
            summary="Home captain fit.",
            direction="home_positive",
            reliability_score=0.8,
            applies_to_market="winner",
        ),
        EvidenceItem(
            evidence_id="e2",
            forecast_id=sample_match.match_id,
            source="skysports.com",
            url="https://skysports.com/1",
            timestamp=datetime.now(timezone.utc),
            summary="Away side missing defender.",
            direction="home_positive",
            reliability_score=0.75,
            applies_to_market="winner",
        ),
    ]

    assert agent._should_stop(evidence[:1], step=1) is False
    assert agent._should_stop(evidence, step=1) is True
    assert agent._should_stop([], step=4) is True


def test_news_context_agent_run_collects_and_saves_evidence(sample_match, sample_baseline) -> None:
    vector_repo = FakeVectorRepository(
        [
            ArticleChunk(
                chunk_id="chunk-1",
                content="Prior report says Arsenal expect their starting striker back.",
                source="theathletic.com",
                url="https://theathletic.com/story",
                published_at="2026-04-24T10:00:00+00:00",
                teams=["Arsenal"],
            )
        ]
    )
    evidence_repo = FakeEvidenceRepository()
    llm = FakeLLM(
        tool_responses=[
            {
                "content": [
                    {"type": "text", "text": "Need fresher team news before stopping."},
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "search_news",
                        "input": {"query": "Arsenal Chelsea team news", "max_results": 3},
                    },
                ]
            },
            {"content": [{"type": "text", "text": "Enough evidence collected for this match."}]},
        ],
        chat_responses=[
            "[]",
            (
                '[{"source":"bbc.co.uk","url":"https://bbc.co.uk/story","summary":"BBC reports Arsenal expect '
                'their first-choice striker to start.","direction":"home_positive","applies_to_market":"winner"}]'
            ),
            (
                '[{"source":"skysports.com","url":"https://skysports.com/story","summary":"Sky Sports says Chelsea '
                'rotated heavily in midweek and may manage minutes.","direction":"home_positive","applies_to_market":"winner"}]'
            ),
        ],
    )
    dispatcher = ToolDispatcher(
        FakeMCPClient(
            result=[
                {
                    "url": "https://bbc.co.uk/story",
                    "title": "Arsenal striker in line to return",
                    "snippet": "Expected to be available against Chelsea.",
                }
            ]
        )
    )
    agent = NewsContextAgent(
        llm=llm,
        tool_dispatcher=dispatcher,
        vector_repo=vector_repo,
        evidence_repo=evidence_repo,
        source_reliability_repo=FakeSourceReliabilityRepository({"bbc.co.uk": 0.85, "skysports.com": 0.85}),
        max_steps=3,
        min_evidence_count=2,
        min_avg_reliability=0.8,
    )

    result = agent.run(_base_state(sample_match, sample_baseline))

    assert len(result["evidence_items"]) == 2
    assert [item.source for item in result["evidence_items"]] == ["bbc.co.uk", "skysports.com"]
    assert all(item.forecast_id == sample_match.match_id for item in result["evidence_items"])
    assert len(result["interpreted_evidence"]) == 2
    assert all(isinstance(item, InterpretedEvidence) for item in result["interpreted_evidence"])
    assert len(evidence_repo.saved) == 2
    assert result["errors"] == []
    assert len(llm.tool_calls) == 2


def test_news_context_agent_ignores_malformed_evidence_json(sample_match, sample_baseline) -> None:
    llm = FakeLLM(
        tool_responses=[{"content": [{"type": "text", "text": "Enough evidence collected for this match."}]}],
        chat_responses=["not-json-at-all"],
    )
    agent = NewsContextAgent(
        llm=llm,
        tool_dispatcher=ToolDispatcher(FakeMCPClient(result=[])),
        vector_repo=FakeVectorRepository([]),
        evidence_repo=FakeEvidenceRepository(),
        source_reliability_repo=FakeSourceReliabilityRepository({}),
        max_steps=1,
        min_evidence_count=1,
        min_avg_reliability=0.5,
    )

    result = agent.run(_base_state(sample_match, sample_baseline))

    assert result["evidence_items"] == []
    assert result["errors"] == []


def test_news_context_agent_records_unknown_tool_error(sample_match, sample_baseline) -> None:
    llm = FakeLLM(
        tool_responses=[
            {
                "content": [
                    {"type": "text", "text": "Need lineup confirmation."},
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "unknown_tool",
                        "input": {"query": "Arsenal lineup"},
                    },
                ]
            }
        ],
        chat_responses=[],
    )
    agent = NewsContextAgent(
        llm=llm,
        tool_dispatcher=ToolDispatcher(FakeMCPClient(result=[])),
        vector_repo=FakeVectorRepository([]),
        evidence_repo=FakeEvidenceRepository(),
        source_reliability_repo=FakeSourceReliabilityRepository({}),
        max_steps=2,
        min_evidence_count=2,
        min_avg_reliability=0.5,
    )

    result = agent.run(_base_state(sample_match, sample_baseline))

    assert result["evidence_items"] == []
    assert result["errors"] == ["Tool dispatch failed for unknown_tool: Unknown tool: unknown_tool"]


def test_news_context_agent_records_evidence_save_failures(sample_match, sample_baseline) -> None:
    llm = FakeLLM(
        tool_responses=[
            {
                "content": [
                    {"type": "text", "text": "Search for a fresh report."},
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "search_news",
                        "input": {"query": "Arsenal Chelsea injuries", "max_results": 1},
                    },
                ]
            }
        ],
        chat_responses=[
            (
                '[{"source":"bbc.co.uk","url":"https://bbc.co.uk/story","summary":"BBC says Arsenal expect key '
                'midfielders to be available.","direction":"home_positive","applies_to_market":"winner"}]'
            )
        ],
    )
    agent = NewsContextAgent(
        llm=llm,
        tool_dispatcher=ToolDispatcher(
            FakeMCPClient(
                result=[
                    {
                        "url": "https://bbc.co.uk/story",
                        "title": "Arsenal midfield boost",
                        "snippet": "Key midfielders expected to return.",
                    }
                ]
            )
        ),
        vector_repo=FakeVectorRepository([]),
        evidence_repo=FailingEvidenceRepository(),
        source_reliability_repo=FakeSourceReliabilityRepository({"bbc.co.uk": 0.85}),
        max_steps=1,
        min_evidence_count=1,
        min_avg_reliability=0.5,
    )

    result = agent.run(_base_state(sample_match, sample_baseline))

    assert len(result["evidence_items"]) == 1
    assert result["errors"]
    assert result["errors"][0].startswith("Failed to save evidence ")


def test_extract_evidence_suppresses_conflicting_market_directions(sample_match, sample_baseline) -> None:
    # LLM returns applies_to_market="winner" but also a non-neutral goals_direction —
    # the extracted InterpretedEvidence must zero out goals_direction to prevent
    # winner-only evidence from moving the totals market.
    conflicting_json = (
        '[{"source":"bbc.co.uk","url":"https://bbc.co.uk/story",'
        '"summary":"Home striker is fit.","direction":"home_positive",'
        '"applies_to_market":"winner","winner_direction":"home_positive","goals_direction":"over_positive"}]'
    )
    llm = FakeLLM(tool_responses=[], chat_responses=[conflicting_json])
    agent = NewsContextAgent(
        llm=llm,
        tool_dispatcher=ToolDispatcher(FakeMCPClient(result=[])),
        vector_repo=FakeVectorRepository([]),
        evidence_repo=FakeEvidenceRepository(),
        source_reliability_repo=FakeSourceReliabilityRepository({"bbc.co.uk": 0.85}),
        max_steps=1,
        min_evidence_count=1,
        min_avg_reliability=0.5,
    )

    _, interp_items = agent._extract_evidence(
        match=sample_match,
        baseline=sample_baseline,
        forecast_id="forecast-1",
        source_material=conflicting_json,
        source_label="search_news",
        max_items=1,
    )

    assert len(interp_items) == 1
    item = interp_items[0]
    assert item.winner_direction == "home_positive"
    assert item.goals_direction == "neutral", "goals_direction must be suppressed for winner-only evidence"
