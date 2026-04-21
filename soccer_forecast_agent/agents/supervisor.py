"""Supervisor agent — builds and runs the LangGraph workflow."""

from typing import TypedDict, Any
from soccer_forecast_agent.models.match import Match, MarketOdds, BaselineForecast, Forecast
from soccer_forecast_agent.models.evidence import EvidenceItem


class GraphState(TypedDict):
    """Shared state passed between all agent nodes in the LangGraph graph."""

    competition: str
    days_ahead: int
    matches: list[Match]
    odds_map: dict[str, MarketOdds]
    baseline_forecasts: dict[str, BaselineForecast]
    current_match_id: str | None
    evidence_items: list[EvidenceItem]
    forecast: Forecast | None
    alert_sent: bool
    errors: list[str]


class SupervisorAgent:
    """Builds the LangGraph graph, registers agent nodes, and defines routing logic."""

    def __init__(self, stats_agent, news_agent, synthesis_agent) -> None:
        """Initialise with injected agent instances — no agent logic lives here."""
        self._stats = stats_agent
        self._news = news_agent
        self._synthesis = synthesis_agent

    def build_graph(self):
        """Construct and compile the LangGraph StateGraph with all nodes and conditional edges."""
        raise NotImplementedError

    def run(self, competition: str, days_ahead: int = 7) -> list[Forecast]:
        """Execute the full workflow and return all forecasts produced in this run."""
        raise NotImplementedError
