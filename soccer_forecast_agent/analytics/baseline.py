"""Baseline strategy protocol and simple form-based implementation."""

from typing import Protocol
from soccer_forecast_agent.models.match import MatchContext, BaselineForecast


class BaselineStrategy(Protocol):
    """Contract for any statistical baseline that produces match probabilities."""

    def compute(self, context: MatchContext) -> BaselineForecast:
        """Compute win/draw/over/under probabilities from match context. Must not use market odds."""
        ...


class SimpleBaselineStrategy:
    """Form and goals-based baseline. No market odds used — baseline must be independent of market prices."""

    RESULT_POINTS = {"W": 3, "D": 1, "L": 0}
    MAX_FORM_POINTS = 15  # 5 wins × 3 points
    WIN_PROBABILITY_FLOOR = 0.05

    def __init__(self, home_advantage_boost: float = 0.05) -> None:
        """Initialise with a configurable home advantage scalar added to the home team's raw strength."""
        self._home_boost = home_advantage_boost

    def compute(self, context: MatchContext) -> BaselineForecast:
        """Derive probabilities from form, home advantage, and goals averages."""
        home_form = self._form_score(context.recent_home_results)
        away_form = self._form_score(context.recent_away_results)

        home_attack = context.home_goals_scored_avg
        home_defense = context.home_goals_conceded_avg
        away_attack = context.away_goals_scored_avg
        away_defense = context.away_goals_conceded_avg

        home_strength = (home_form + home_attack) / (1 + home_defense) + self._home_boost
        away_strength = (away_form + away_attack) / (1 + away_defense)

        total = home_strength + away_strength
        raw_home = home_strength / total
        raw_away = away_strength / total

        draw_weight = 0.26
        available_mass = 1 - draw_weight
        raw_home_prob = raw_home * available_mass
        raw_away_prob = raw_away * available_mass
        if raw_away_prob < self.WIN_PROBABILITY_FLOOR:
            away_win = self.WIN_PROBABILITY_FLOOR
            home_win = available_mass - away_win
        elif raw_home_prob < self.WIN_PROBABILITY_FLOOR:
            home_win = self.WIN_PROBABILITY_FLOOR
            away_win = available_mass - home_win
        else:
            home_win, away_win = raw_home_prob, raw_away_prob
        draw = draw_weight

        total_goals_est = (home_attack + away_defense + away_attack + home_defense) / 2
        over = min(max(total_goals_est / 4.5, 0.1), 0.9)
        under = 1.0 - over

        return BaselineForecast(
            home_win=round(home_win, 4),
            draw=round(draw, 4),
            away_win=round(away_win, 4),
            over_2_5=round(over, 4),
            under_2_5=round(under, 4),
        )

    def _form_score(self, results: list[str]) -> float:
        """Normalise last-N results to a 0–1 form score."""
        if not results:
            return 0.5
        points = sum(self.RESULT_POINTS.get(r, 0) for r in results)
        return points / (len(results) * 3)

