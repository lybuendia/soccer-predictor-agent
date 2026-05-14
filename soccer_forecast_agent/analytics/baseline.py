"""Baseline strategy protocol and implementations."""

from math import exp, factorial
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


class EnhancedBaselineStrategy:
    """Improved baseline: better goals model and recency-weighted form.

    Two weaknesses of SimpleBaselineStrategy are addressed:
    1. Goals model — the original formula added all four goal averages (attack + conceded for
       both teams), double-counting and inflating the over-probability above random.
       The fix averages each team's attack with the opponent's conceded rate (proper
       expected-goals estimate), then blends 70 / 30 toward the league average to prevent
       noisy 5-game windows from producing extreme predictions.
    2. Form recency — results are returned newest-first by get_recent_finished, so
       index 0 receives weight 1.0 and each older result decays by 0.75 per step.
       Draw probability is kept at a fixed 0.26 — the dynamic draw variant increased
       Brier score by adding miscalibrated variance.
    """

    LEAGUE_GOALS_PER_TEAM: float = 1.35  # EPL average goals per team per game
    LEAGUE_GOALS_TOTAL: float = 2 * 1.35  # 2.70 per match
    GOALS_DIVISOR: float = 4.9           # calibrated so league-avg total → ~55 % over
    DRAW_WEIGHT: float = 0.26
    WIN_PROBABILITY_FLOOR: float = 0.05

    def __init__(self, home_advantage_boost: float = 0.05) -> None:
        """Initialise with a configurable home advantage scalar."""
        self._home_boost = home_advantage_boost

    def compute(self, context: MatchContext) -> BaselineForecast:
        """Derive probabilities with fixed draw, recency form, and blended expected goals."""
        home_form = self._form_score(context.recent_home_results)
        away_form = self._form_score(context.recent_away_results)

        home_attack = context.home_goals_scored_avg
        away_attack = context.away_goals_scored_avg
        home_defense = context.home_goals_conceded_avg
        away_defense = context.away_goals_conceded_avg

        home_strength = (home_form + home_attack) / (1 + home_defense) + self._home_boost
        away_strength = (away_form + away_attack) / (1 + away_defense)

        total = home_strength + away_strength or 1.0
        raw_home = home_strength / total
        raw_away = away_strength / total

        available = 1.0 - self.DRAW_WEIGHT
        home_win = max(raw_home * available, self.WIN_PROBABILITY_FLOOR)
        away_win = max(raw_away * available, self.WIN_PROBABILITY_FLOOR)
        hw_total = home_win + away_win
        if hw_total > available:
            home_win = home_win / hw_total * available
            away_win = away_win / hw_total * available
        draw = 1.0 - home_win - away_win

        over = self._over_probability(home_attack, home_defense, away_attack, away_defense)

        return BaselineForecast(
            home_win=round(home_win, 4),
            draw=round(draw, 4),
            away_win=round(away_win, 4),
            over_2_5=round(over, 4),
            under_2_5=round(1.0 - over, 4),
        )

    def _over_probability(
        self,
        h_att: float,
        h_def: float,
        a_att: float,
        a_def: float,
    ) -> float:
        """P(total goals > 2.5) from blended expected goals.

        Each team's expected goals = average of their attack rate and their opponent's
        concede rate, which properly adjusts for opponent quality without double-counting.
        A 70/30 blend toward the league average prevents noisy 5-game windows from
        producing extreme predictions.
        """
        exp_home = (h_att + a_def) / 2
        exp_away = (a_att + h_def) / 2
        raw_total = exp_home + exp_away + self._home_boost * 0.3
        blended = 0.70 * raw_total + 0.30 * self.LEAGUE_GOALS_TOTAL
        return min(max(blended / self.GOALS_DIVISOR, 0.25), 0.75)

    def _form_score(self, results: list[str]) -> float:
        """Recency-weighted form — results are newest-first; index 0 has weight 1.0, decays by 0.75."""
        if not results:
            return 0.5
        points_map = {"W": 3, "D": 1, "L": 0}
        weights = [0.75 ** i for i in range(len(results))]
        total_w = sum(weights)
        score = sum(points_map.get(r, 0) * w for r, w in zip(results, weights))
        return score / (total_w * 3)

