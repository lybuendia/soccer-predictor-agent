"""Feature extraction — builds a MatchContext from raw match data and historical records."""

from soccer_forecast_agent.models.match import Match, MarketOdds, MatchContext


class FeatureExtractor:
    """Transforms raw match and historical data into a typed MatchContext for the baseline strategy."""

    def extract(
        self,
        match: Match,
        odds: MarketOdds,
        home_results: list[str],
        away_results: list[str],
        home_goals_scored: list[float],
        home_goals_conceded: list[float],
        away_goals_scored: list[float],
        away_goals_conceded: list[float],
    ) -> MatchContext:
        """Build a MatchContext by averaging goals series and attaching recent form."""
        return MatchContext(
            match=match,
            odds=odds,
            recent_home_results=home_results[-5:],
            recent_away_results=away_results[-5:],
            home_goals_scored_avg=self._avg(home_goals_scored),
            home_goals_conceded_avg=self._avg(home_goals_conceded),
            away_goals_scored_avg=self._avg(away_goals_scored),
            away_goals_conceded_avg=self._avg(away_goals_conceded),
        )

    def _avg(self, values: list[float]) -> float:
        """Return the mean of a list, or 1.2 (league average) if empty."""
        return sum(values) / len(values) if values else 1.2
