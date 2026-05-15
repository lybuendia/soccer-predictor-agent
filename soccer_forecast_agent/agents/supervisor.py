"""Supervisor agent — builds and runs the LangGraph workflow."""

from typing import TypedDict
from uuid import uuid4
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

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
    current_forecast_id: str | None
    evidence_items: list[EvidenceItem]
    interpreted_evidence: list[InterpretedEvidence]
    forecast: Forecast | None
    all_forecasts: list[Forecast]
    alert_sent: bool
    errors: list[str]


class SupervisorAgent:
    """Builds the LangGraph graph, registers agent nodes, and defines routing logic."""

    def __init__(
        self,
        stats_agent,
        news_agent,
        synthesis_agent,
        min_edge_threshold: float = 0.05,
    ) -> None:
        """Initialise with injected agent instances and the minimum baseline edge required to trigger research."""
        self._stats = stats_agent
        self._news = news_agent
        self._synthesis = synthesis_agent
        self._min_edge_threshold = min_edge_threshold

    def build_graph(self) -> CompiledStateGraph:
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
            self._route_after_setup,
            {"research": "news_context", "synthesise": "synthesis", "done": END},
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
            "current_forecast_id": None,
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
            return {**state, "current_match_id": None, "current_forecast_id": None}
        return {
            **state,
            "current_match_id": pending[0],
            "current_forecast_id": str(uuid4()),
            "pending_match_ids": pending[1:],
            "evidence_items": [],
            "interpreted_evidence": [],
            "forecast": None,
        }

    def _route_after_setup(self, state: GraphState) -> str:
        """Route to 'research' if baseline edge clears the threshold, 'synthesise' to skip research, 'done' when finished."""
        if state.get("current_match_id") is None:
            return "done"
        match = next(
            (m for m in state.get("matches", []) if m.match_id == state["current_match_id"]),
            None,
        )
        label = f"{match.home_team} vs {match.away_team}" if match else state["current_match_id"]
        if self._has_baseline_edge(state):
            print(f"[EDGE]  {label} — baseline edge detected, running research")
            return "research"
        print(f"[SKIP]  {label} — no baseline edge, skipping research")
        return "synthesise"

    def _has_baseline_edge(self, state: GraphState) -> bool:
        """Return True if the baseline shows at least one market edge above the minimum threshold."""
        match_id = state.get("current_match_id")
        baseline = state.get("baseline_forecasts", {}).get(match_id)
        odds = state.get("odds_map", {}).get(match_id)
        if baseline is None or odds is None:
            return True  # can't determine without data — proceed with research

        def implied(o: float) -> float:
            return 1 / o if o > 0 else 0.0

        h, d, a = implied(odds.home_win), implied(odds.draw), implied(odds.away_win)
        total_w = h + d + a or 1.0
        o, u = implied(odds.over_2_5), implied(odds.under_2_5)
        total_g = o + u or 1.0

        max_edge = max(
            baseline.home_win - h / total_w,
            baseline.draw - d / total_w,
            baseline.away_win - a / total_w,
            baseline.over_2_5 - o / total_g,
            baseline.under_2_5 - u / total_g,
        )
        return max_edge >= self._min_edge_threshold
