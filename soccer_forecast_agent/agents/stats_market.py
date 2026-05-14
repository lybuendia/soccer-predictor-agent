"""Stats and Market agent — fetches fixtures, odds, and computes the statistical baseline."""

from soccer_forecast_agent.agents.supervisor import GraphState
from soccer_forecast_agent.analytics.baseline import BaselineStrategy
from soccer_forecast_agent.analytics.features import FeatureExtractor
from soccer_forecast_agent.memory.repository import MatchRepository
from soccer_forecast_agent.tools.fixtures import FixtureFetcher
from soccer_forecast_agent.tools.odds import OddsFetcher


class StatsMarketAgent:
    """Fetches match data and computes a baseline forecast for each upcoming fixture. Does not call the LLM."""

    def __init__(
        self,
        fixture_fetcher: FixtureFetcher,
        odds_fetcher: OddsFetcher,
        baseline_strategy: BaselineStrategy,
        feature_extractor: FeatureExtractor,
        match_repo: MatchRepository,
    ) -> None:
        """Initialise with all injected data dependencies."""
        self._fixtures = fixture_fetcher
        self._odds = odds_fetcher
        self._baseline = baseline_strategy
        self._features = feature_extractor
        self._match_repo = match_repo

    def run(self, state: GraphState) -> GraphState:
        """Fetch fixtures and odds, compute baseline forecasts, and update shared state."""
        competition = state["competition"]
        days_ahead = state["days_ahead"]
        matches = self._fixtures.fetch_upcoming(competition=competition, days_ahead=days_ahead)

        odds_map = dict(state.get("odds_map", {}))
        baseline_forecasts = dict(state.get("baseline_forecasts", {}))
        errors = list(state.get("errors", []))

        for match in matches:
            self._match_repo.save_match(match)
            odds = self._odds.fetch_odds(match.home_team, match.away_team)
            if odds is None:
                errors.append(f"No odds found for {match.home_team} vs {match.away_team}")
                continue

            home_history = self._match_repo.get_recent_finished(match.home_team, competition=competition, limit=5)
            away_history = self._match_repo.get_recent_finished(match.away_team, competition=competition, limit=5)

            context = self._features.extract(
                match=match,
                odds=odds,
                home_results=self._results_for_team(match.home_team, home_history),
                away_results=self._results_for_team(match.away_team, away_history),
                home_goals_scored=self._goals_scored_for_team(match.home_team, home_history),
                home_goals_conceded=self._goals_conceded_for_team(match.home_team, home_history),
                away_goals_scored=self._goals_scored_for_team(match.away_team, away_history),
                away_goals_conceded=self._goals_conceded_for_team(match.away_team, away_history),
            )
            baseline_forecasts[match.match_id] = self._baseline.compute(context)
            odds_map[match.match_id] = odds

        return {
            **state,
            "matches": matches,
            "odds_map": odds_map,
            "baseline_forecasts": baseline_forecasts,
            "pending_match_ids": list(baseline_forecasts.keys()),
            "errors": errors,
        }

    def _results_for_team(self, team: str, matches: list) -> list[str]:
        """Return W/D/L results for a team from recent finished matches."""
        return [self._result_for_team(team, match) for match in matches]

    def _goals_scored_for_team(self, team: str, matches: list) -> list[float]:
        """Return goals scored by a team from recent finished matches."""
        return [float(self._score_tuple_for_team(team, match)[0]) for match in matches]

    def _goals_conceded_for_team(self, team: str, matches: list) -> list[float]:
        """Return goals conceded by a team from recent finished matches."""
        return [float(self._score_tuple_for_team(team, match)[1]) for match in matches]

    def _result_for_team(self, team: str, match) -> str:
        """Return W, D, or L for a team from a finished match."""
        goals_for, goals_against = self._score_tuple_for_team(team, match)
        if goals_for > goals_against:
            return "W"
        if goals_for < goals_against:
            return "L"
        return "D"

    def _score_tuple_for_team(self, team: str, match) -> tuple[int, int]:
        """Return (goals_for, goals_against) for a team from a finished match."""
        if match.final_score is None:
            raise ValueError(f"Finished match {match.match_id} has no final score")

        home_goals_str, away_goals_str = match.final_score.split("-", maxsplit=1)
        home_goals = int(home_goals_str)
        away_goals = int(away_goals_str)
        if match.home_team == team:
            return home_goals, away_goals
        return away_goals, home_goals
