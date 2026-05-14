"""Brier score computation for forecast calibration evaluation."""

from dataclasses import dataclass

from soccer_forecast_agent.models.match import BaselineForecast


@dataclass(frozen=True)
class MatchOutcome:
    """Actual match result expressed as binary indicator vectors."""

    home_win: bool
    draw: bool
    away_win: bool
    over_2_5: bool
    under_2_5: bool


@dataclass(frozen=True)
class BrierScores:
    """Mean Brier scores aggregated across a set of evaluated matches."""

    winner_brier: float
    goals_brier: float
    n_matches: int


def outcome_from_score(final_score: str) -> MatchOutcome:
    """Parse a 'home-away' score string into a MatchOutcome."""
    home_str, away_str = final_score.split("-", 1)
    home, away = int(home_str), int(away_str)
    total = home + away
    return MatchOutcome(
        home_win=home > away,
        draw=home == away,
        away_win=away > home,
        over_2_5=total > 2,
        under_2_5=total <= 2,
    )


def brier_score_winner(forecast: BaselineForecast, outcome: MatchOutcome) -> float:
    """Brier score for the 1X2 winner market — average squared error across three outcomes."""
    return (
        (forecast.home_win - float(outcome.home_win)) ** 2
        + (forecast.draw - float(outcome.draw)) ** 2
        + (forecast.away_win - float(outcome.away_win)) ** 2
    ) / 3


def brier_score_goals(forecast: BaselineForecast, outcome: MatchOutcome) -> float:
    """Brier score for the over/under 2.5 goals market — average squared error across two outcomes."""
    return (
        (forecast.over_2_5 - float(outcome.over_2_5)) ** 2
        + (forecast.under_2_5 - float(outcome.under_2_5)) ** 2
    ) / 2


def aggregate(winner_scores: list[float], goals_scores: list[float]) -> BrierScores:
    """Return mean Brier scores across all evaluated matches."""
    n = len(winner_scores)
    if n == 0:
        return BrierScores(winner_brier=0.0, goals_brier=0.0, n_matches=0)
    return BrierScores(
        winner_brier=round(sum(winner_scores) / n, 4),
        goals_brier=round(sum(goals_scores) / n, 4),
        n_matches=n,
    )
