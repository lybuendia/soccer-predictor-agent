from datetime import datetime, timedelta, timezone

import pytest

from soccer_forecast_agent.models.evidence import EvidenceItem
from soccer_forecast_agent.models.match import BaselineForecast, Forecast, Match, MarketOdds


@pytest.fixture
def sample_match() -> Match:
    return Match(
        match_id="match-1",
        competition="PL",
        home_team="Arsenal",
        away_team="Chelsea",
        kickoff_time=datetime(2026, 5, 1, 19, 0, tzinfo=timezone.utc),
        status="upcoming",
    )


@pytest.fixture
def sample_odds() -> MarketOdds:
    return MarketOdds(
        odds_id="odds-1",
        match_id="match-1",
        timestamp=datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc),
        home_win=1.9,
        draw=3.6,
        away_win=4.0,
        over_2_5=1.95,
        under_2_5=1.85,
    )


@pytest.fixture
def sample_baseline() -> BaselineForecast:
    return BaselineForecast(
        home_win=0.48,
        draw=0.26,
        away_win=0.26,
        over_2_5=0.55,
        under_2_5=0.45,
    )


@pytest.fixture
def sample_forecast(sample_baseline: BaselineForecast) -> Forecast:
    return Forecast(
        forecast_id="forecast-1",
        match_id="match-1",
        run_timestamp=datetime(2026, 4, 25, 12, 30, tzinfo=timezone.utc),
        baseline=sample_baseline,
        adjusted_home_win=0.5,
        adjusted_draw=0.25,
        adjusted_away_win=0.25,
        adjusted_over_2_5=0.57,
        adjusted_under_2_5=0.43,
        confidence_score=0.72,
        edge_market="home_win",
        edge_value=0.08,
        alert_sent=False,
        rationale="Home side has stronger form and better recent attacking output.",
    )


@pytest.fixture
def recent_evidence() -> list[EvidenceItem]:
    now = datetime.now(timezone.utc)
    return [
        EvidenceItem(
            evidence_id="e1",
            forecast_id="forecast-1",
            source="bbc.co.uk",
            url="https://bbc.co.uk/1",
            timestamp=now - timedelta(hours=1),
            summary="Positive squad availability for the home team.",
            direction="home_positive",
            reliability_score=0.8,
            applies_to_market="winner",
        ),
        EvidenceItem(
            evidence_id="e2",
            forecast_id="forecast-1",
            source="premierleague.com",
            url="https://premierleague.com/1",
            timestamp=now - timedelta(hours=2),
            summary="Official training update confirms expected starters.",
            direction="home_positive",
            reliability_score=0.9,
            applies_to_market="winner",
        ),
        EvidenceItem(
            evidence_id="e3",
            forecast_id="forecast-1",
            source="skysports.com",
            url="https://skysports.com/1",
            timestamp=now - timedelta(hours=3),
            summary="Match preview points to sustained attacking form.",
            direction="neutral",
            reliability_score=0.7,
            applies_to_market="both",
        ),
    ]
