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
        raise NotImplementedError
