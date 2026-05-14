"""Data models for matches, odds, forecasts, and related structures."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Match:
    """A Premier League fixture with its current status and optional final score."""

    match_id: str
    competition: str
    home_team: str
    away_team: str
    kickoff_time: datetime
    status: str  # "upcoming" | "live" | "resolved"
    final_score: str | None = None


@dataclass
class MarketOdds:
    """Decimal odds for a match across winner and over/under markets at a point in time."""

    odds_id: str
    match_id: str
    timestamp: datetime
    home_win: float
    draw: float
    away_win: float
    over_2_5: float
    under_2_5: float


@dataclass
class BaselineForecast:
    """Probabilities produced by the statistical baseline, independent of market odds."""

    home_win: float
    draw: float
    away_win: float
    over_2_5: float
    under_2_5: float


@dataclass
class Forecast:
    """Complete forecast record for a match, including baseline, adjusted probabilities, and alert state."""

    forecast_id: str
    match_id: str
    run_timestamp: datetime
    baseline: BaselineForecast
    adjusted_home_win: float | None
    adjusted_draw: float | None
    adjusted_away_win: float | None
    adjusted_over_2_5: float | None
    adjusted_under_2_5: float | None
    confidence_score: float
    edge_market: str | None
    edge_value: float | None
    alert_sent: bool
    rationale: str


@dataclass
class MatchContext:
    """All inputs required by the baseline strategy to compute a forecast."""

    match: Match
    odds: MarketOdds
    recent_home_results: list[str]
    recent_away_results: list[str]
    home_goals_scored_avg: float
    home_goals_conceded_avg: float
    away_goals_scored_avg: float
    away_goals_conceded_avg: float


@dataclass
class SynthesisResult:
    """Typed intermediate output of synthesis math, used to assemble the final Forecast."""

    adjusted: BaselineForecast
    confidence_score: float
    edge_market: str | None
    edge_value: float | None


@dataclass
class AlertPayload:
    """Everything needed to render and send an alert email."""

    match: Match
    forecast: Forecast
    rationale: str
    disclaimer: str = (
        "This is decision support only. Not a guarantee of profit. "
        "Probabilities are estimates, not certainties."
    )
