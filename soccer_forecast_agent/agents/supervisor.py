"""Supervisor agent — builds and runs the LangGraph workflow."""

from typing import TypedDict
from langgraph.graph import END, START, StateGraph

from soccer_forecast_agent.models.match import Match, MarketOdds, BaselineForecast, Forecast
from soccer_forecast_agent.models.evidence import EvidenceItem, InterpretedEvidence


class GraphState(TypedDict):
    """Shared state passed between all agent nodes in the LangGraph graph."""

    competition: str
    days_ahead: int
    matches: list[Match]
    odds_map: dict[str, MarketOdds]
    baseline_forecasts: dict[str, BaselineForecast]
    pending_match_ids: list[str]
    current_match_id: str | None
    evidence_items: list[EvidenceItem]
    interpreted_evidence: list[InterpretedEvidence]
    forecast: Forecast | None
    all_forecasts: list[Forecast]
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
        graph = StateGraph(GraphState)

        graph.add_node("stats_market", self._stats.run)
        graph.add_node("setup_next_match", self._setup_next_match)
        graph.add_node("news_context", self._news.run)
        graph.add_node("synthesis", self._synthesis.run)

        graph.add_edge(START, "stats_market")
        graph.add_edge("stats_market", "setup_next_match")
        graph.add_conditional_edges(
            "setup_next_match",
            self._route_next,
            {"next": "news_context", "done": END},
        )
        graph.add_edge("news_context", "synthesis")
        graph.add_edge("synthesis", "setup_next_match")

        return graph.compile()

    def run(self, competition: str, days_ahead: int = 7) -> list[Forecast]:
        """Execute the full workflow and return all forecasts produced in this run."""
        graph = self.build_graph()
        initial_state: GraphState = {
            "competition": competition,
            "days_ahead": days_ahead,
            "matches": [],
            "odds_map": {},
            "baseline_forecasts": {},
            "pending_match_ids": [],
            "current_match_id": None,
            "evidence_items": [],
            "interpreted_evidence": [],
            "forecast": None,
            "all_forecasts": [],
            "alert_sent": False,
            "errors": [],
        }
        final_state = graph.invoke(initial_state)
        for err in final_state.get("errors", []):
            print(f"[WARN] {err}")
        return final_state.get("all_forecasts", [])

    def _setup_next_match(self, state: GraphState) -> GraphState:
        """Pop the next pending match and reset per-match evidence so each match starts clean."""
        pending = list(state.get("pending_match_ids", []))
        if not pending:
            return {**state, "current_match_id": None}
        return {
            **state,
            "current_match_id": pending[0],
            "pending_match_ids": pending[1:],
            "evidence_items": [],
            "interpreted_evidence": [],
            "forecast": None,
        }

    def _route_next(self, state: GraphState) -> str:
        """Return 'next' if a match is ready to process, 'done' when all matches are exhausted."""
        return "done" if state.get("current_match_id") is None else "next"
