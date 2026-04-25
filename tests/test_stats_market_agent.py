import pytest

from dataclasses import dataclass

from soccer_forecast_agent.agents.stats_market import StatsMarketAgent
from soccer_forecast_agent.analytics.baseline import SimpleBaselineStrategy
from soccer_forecast_agent.analytics.features import FeatureExtractor


@dataclass
class FakeFixtureFetcher:
    matches: list

    def fetch_upcoming(self, competition: str, days_ahead: int) -> list:
        return self.matches


@dataclass
class FakeOddsFetcher:
    odds_by_pair: dict[tuple[str, str], object | None]

    def fetch_odds(self, home_team: str, away_team: str):
        return self.odds_by_pair.get((home_team, away_team))


class FakeMatchRepository:
    def __init__(self) -> None:
        self.saved_matches = []
        self.history_by_team = {}

    def save_match(self, match) -> None:
        self.saved_matches.append(match)

    def get_recent_finished(self, team: str, competition: str, limit: int = 5) -> list:
        return self.history_by_team.get(team, [])[:limit]


def _base_state() -> dict:
    return {
        "competition": "PL",
        "days_ahead": 7,
        "matches": [],
        "odds_map": {},
        "baseline_forecasts": {},
        "current_match_id": None,
        "evidence_items": [],
        "forecast": None,
        "alert_sent": False,
        "errors": [],
    }


def test_stats_market_agent_populates_matches_odds_and_baselines(sample_match, sample_odds):
    fixture_fetcher = FakeFixtureFetcher(matches=[sample_match])
    odds_fetcher = FakeOddsFetcher({("Arsenal", "Chelsea"): sample_odds})
    match_repo = FakeMatchRepository()
    agent = StatsMarketAgent(
        fixture_fetcher=fixture_fetcher,
        odds_fetcher=odds_fetcher,
        baseline_strategy=SimpleBaselineStrategy(),
        feature_extractor=FeatureExtractor(),
        match_repo=match_repo,
    )

    result = agent.run(_base_state())

    assert result["matches"] == [sample_match]
    assert result["odds_map"]["match-1"] == sample_odds
    assert "match-1" in result["baseline_forecasts"]
    total = (
        result["baseline_forecasts"]["match-1"].home_win
        + result["baseline_forecasts"]["match-1"].draw
        + result["baseline_forecasts"]["match-1"].away_win
    )
    assert total == pytest.approx(1.0, abs=1e-4)
    assert match_repo.saved_matches == [sample_match]
    assert result["errors"] == []


def test_stats_market_agent_records_error_when_odds_missing(sample_match):
    fixture_fetcher = FakeFixtureFetcher(matches=[sample_match])
    odds_fetcher = FakeOddsFetcher({("Arsenal", "Chelsea"): None})
    match_repo = FakeMatchRepository()
    agent = StatsMarketAgent(
        fixture_fetcher=fixture_fetcher,
        odds_fetcher=odds_fetcher,
        baseline_strategy=SimpleBaselineStrategy(),
        feature_extractor=FeatureExtractor(),
        match_repo=match_repo,
    )

    result = agent.run(_base_state())

    assert result["matches"] == [sample_match]
    assert result["odds_map"] == {}
    assert result["baseline_forecasts"] == {}
    assert result["errors"] == ["No odds found for Arsenal vs Chelsea"]


def test_stats_market_agent_uses_recent_finished_history_for_baseline(sample_match, sample_odds):
    fixture_fetcher = FakeFixtureFetcher(matches=[sample_match])
    odds_fetcher = FakeOddsFetcher({("Arsenal", "Chelsea"): sample_odds})
    match_repo = FakeMatchRepository()
    match_repo.history_by_team = {
        "Arsenal": [
            sample_match.__class__(
                match_id="h1",
                competition="PL",
                home_team="Arsenal",
                away_team="Spurs",
                kickoff_time=sample_match.kickoff_time,
                status="resolved",
                final_score="3-1",
            ),
            sample_match.__class__(
                match_id="h2",
                competition="PL",
                home_team="Liverpool",
                away_team="Arsenal",
                kickoff_time=sample_match.kickoff_time,
                status="resolved",
                final_score="0-2",
            ),
        ],
        "Chelsea": [
            sample_match.__class__(
                match_id="h3",
                competition="PL",
                home_team="Chelsea",
                away_team="Everton",
                kickoff_time=sample_match.kickoff_time,
                status="resolved",
                final_score="1-1",
            ),
            sample_match.__class__(
                match_id="h4",
                competition="PL",
                home_team="Villa",
                away_team="Chelsea",
                kickoff_time=sample_match.kickoff_time,
                status="resolved",
                final_score="2-0",
            ),
        ],
    }
    agent = StatsMarketAgent(
        fixture_fetcher=fixture_fetcher,
        odds_fetcher=odds_fetcher,
        baseline_strategy=SimpleBaselineStrategy(),
        feature_extractor=FeatureExtractor(),
        match_repo=match_repo,
    )

    result = agent.run(_base_state())
    baseline = result["baseline_forecasts"]["match-1"]

    assert baseline.home_win > baseline.away_win
