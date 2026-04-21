"""Synthesis and Alert agent — combines baseline and evidence, applies log-odds adjustment, decides alert."""

import math
from soccer_forecast_agent.agents.supervisor import GraphState
from soccer_forecast_agent.guardrails.alert_guard import AlertGuard
from soccer_forecast_agent.memory.repository import ForecastRepository
from soccer_forecast_agent.models.match import Forecast
from soccer_forecast_agent.providers.llm import LLMProvider
from soccer_forecast_agent.tools.alert_channel import AlertChannel


class SynthesisAlertAgent:
    """Combines baseline probabilities with qualitative evidence via log-odds adjustment and fires alerts."""

    MARKET_RELEVANCE_WEIGHTS = {"both": 1.0, "winner": 0.7, "goals": 0.7}
    DIRECTION_SIGNS = {"home_positive": 1.0, "away_positive": -1.0, "neutral": 0.0, "uncertainty": 0.0}

    def __init__(
        self,
        llm: LLMProvider,
        alert_channel: AlertChannel,
        forecast_repo: ForecastRepository,
        guard: AlertGuard,
        base_sensitivity: float = 0.15,
    ) -> None:
        """Initialise with injected LLM, alert channel, persistence, and guardrail."""
        self._llm = llm
        self._alert = alert_channel
        self._forecast_repo = forecast_repo
        self._guard = guard
        self._base_sensitivity = base_sensitivity

    def run(self, state: GraphState) -> GraphState:
        """Apply log-odds adjustment, evaluate the guard, send alert if thresholds pass, persist forecast."""
        raise NotImplementedError

    def _adjust_probability(self, baseline_p: float, evidence) -> float:
        """Shift baseline_p in log-odds space by the sum of signed evidence deltas, then sigmoid back."""
        log_odds = math.log(baseline_p / (1 - baseline_p))
        delta = sum(
            self.DIRECTION_SIGNS.get(e.direction, 0.0)
            * e.reliability_score
            * self.MARKET_RELEVANCE_WEIGHTS.get(e.applies_to_market, 0.7)
            * self._base_sensitivity
            for e in evidence
        )
        adjusted = 1 / (1 + math.exp(-(log_odds + delta)))
        return round(adjusted, 4)

    def _implied_probability(self, odds: float) -> float:
        """Convert decimal odds to implied probability, without overround correction."""
        return round(1 / odds, 4) if odds > 0 else 0.0
