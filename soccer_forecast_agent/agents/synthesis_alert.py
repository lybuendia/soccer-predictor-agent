"""Synthesis and Alert agent — combines baseline and evidence, applies log-odds adjustment, decides alert."""

import dataclasses
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from soccer_forecast_agent.agents.supervisor import GraphState
from soccer_forecast_agent.guardrails.alert_guard import AlertGuard
from soccer_forecast_agent.models.evidence import InterpretedEvidence
from soccer_forecast_agent.memory.repository import ForecastRepository
from soccer_forecast_agent.models.match import AlertPayload, BaselineForecast, Forecast, Match, MarketOdds, SynthesisResult
from soccer_forecast_agent.prompts.loader import render_synthesis_rationale_messages
from soccer_forecast_agent.providers.llm import LLMProvider
from soccer_forecast_agent.tools.alert_channel import AlertChannel


@dataclass(frozen=True)
class SynthesisTuning:
    """Typed tuning values for probability adjustment, confidence scoring, and rationale generation."""

    base_sensitivity: float = 0.15
    probability_floor: float = 0.01
    probability_ceiling: float = 0.99
    implied_probability_default: float = 0.0
    prior_odds_delta_default: float = 0.0
    source_bonus_per_unique_source: float = 0.05
    source_bonus_cap: float = 0.15
    count_bonus_per_item: float = 0.03
    count_bonus_cap: float = 0.15
    confidence_base: float = 0.35
    confidence_empty_evidence: float = 0.30
    confidence_reliability_weight: float = 0.40
    confidence_cap: float = 0.95
    rounding_precision: int = 4
    rationale_evidence_limit: int = 5
    rationale_temperature: float = 0.0
    rationale_max_tokens: int = 180
    rationale_sentence_limit: int = 3


class SynthesisAlertAgent:
    """Combines baseline probabilities with qualitative evidence via log-odds adjustment and fires alerts."""

    # draw_positive evidence reduces both home and away in log-odds space so that
    # raw_draw stays constant while winner_total shrinks, making draw's normalized
    # share rise after renormalization — the only way to lift draw without a direct
    # draw log-odds adjustment.
    WINNER_SIGNS_HOME = {"home_positive": 1.0, "away_positive": -1.0, "draw_positive": -0.5, "neutral": 0.0, "uncertainty": 0.0}
    WINNER_SIGNS_AWAY = {"home_positive": -1.0, "away_positive": 1.0, "draw_positive": -0.5, "neutral": 0.0, "uncertainty": 0.0}
    GOALS_SIGNS_OVER = {"over_positive": 1.0, "under_positive": -1.0, "neutral": 0.0, "uncertainty": 0.0}
    GOALS_SIGNS_UNDER = {"over_positive": -1.0, "under_positive": 1.0, "neutral": 0.0, "uncertainty": 0.0}

    def __init__(
        self,
        llm: LLMProvider,
        alert_channel: AlertChannel,
        forecast_repo: ForecastRepository,
        guard: AlertGuard,
        tuning: SynthesisTuning | None = None,
    ) -> None:
        """Initialise with injected LLM, alert channel, persistence, and guardrail."""
        self._llm = llm
        self._alert = alert_channel
        self._forecast_repo = forecast_repo
        self._guard = guard
        self._tuning = tuning or SynthesisTuning()

    def run(self, state: GraphState) -> GraphState:
        """Apply log-odds adjustment, evaluate the guard, send alert if thresholds pass, persist forecast."""
        current_match_id = state.get("current_match_id")
        errors = list(state.get("errors", []))
        if not current_match_id:
            errors.append("SynthesisAlertAgent requires current_match_id in state")
            return {**state, "errors": errors}

        match = self._match_from_state(state, current_match_id)
        if match is None:
            errors.append(f"Match not found for current_match_id={current_match_id}")
            return {**state, "errors": errors}

        baseline = state.get("baseline_forecasts", {}).get(current_match_id)
        if baseline is None:
            errors.append(f"Baseline forecast not found for match_id={current_match_id}")
            return {**state, "errors": errors}

        odds = state.get("odds_map", {}).get(current_match_id)
        if odds is None:
            errors.append(f"Market odds not found for match_id={current_match_id}")
            return {**state, "errors": errors}

        evidence = list(state.get("evidence_items", []))
        interpreted = list(state.get("interpreted_evidence", []))
        synthesis = self._synthesise(baseline=baseline, interpreted=interpreted, odds=odds)
        rationale = self._build_rationale(
            match=match,
            baseline=baseline,
            adjusted=synthesis.adjusted,
            interpreted=interpreted,
            edge_market=synthesis.edge_market,
        )

        forecast_id = str(uuid4())
        preliminary = Forecast(
            forecast_id=forecast_id,
            match_id=current_match_id,
            run_timestamp=datetime.now(timezone.utc),
            baseline=baseline,
            adjusted_home_win=synthesis.adjusted.home_win,
            adjusted_draw=synthesis.adjusted.draw,
            adjusted_away_win=synthesis.adjusted.away_win,
            adjusted_over_2_5=synthesis.adjusted.over_2_5,
            adjusted_under_2_5=synthesis.adjusted.under_2_5,
            confidence_score=synthesis.confidence_score,
            edge_market=synthesis.edge_market,
            edge_value=synthesis.edge_value,
            alert_sent=False,
            rationale=rationale,
        )

        prior_forecast = self._forecast_repo.get_latest(current_match_id)
        guard_result = self._guard.check(
            forecast=preliminary,
            evidence=evidence,
            prior_alert_at=prior_forecast.run_timestamp if prior_forecast and prior_forecast.alert_sent else None,
            odds_delta=self._tuning.prior_odds_delta_default,
        )

        final_rationale = (
            self._append_guard_reasons(rationale, guard_result.reasons)
            if not guard_result.passed
            else rationale
        )

        # Build the corrected forecast before sending so the payload and persisted record are consistent.
        forecast = dataclasses.replace(preliminary, rationale=final_rationale)

        alert_sent = False
        if guard_result.passed:
            alert_sent = self._alert.send(
                AlertPayload(
                    match=match,
                    forecast=forecast,
                    rationale=final_rationale,
                )
            )
            if not alert_sent:
                errors.append(f"Alert delivery failed for match_id={current_match_id}")

        forecast = dataclasses.replace(forecast, alert_sent=alert_sent)
        self._forecast_repo.save_forecast(forecast)
        all_forecasts = list(state.get("all_forecasts", [])) + [forecast]
        return {
            **state,
            "forecast": forecast,
            "all_forecasts": all_forecasts,
            "alert_sent": alert_sent,
            "errors": errors,
        }

    def _synthesise(self, baseline: BaselineForecast, interpreted: list[InterpretedEvidence], odds: MarketOdds) -> SynthesisResult:
        """Run all synthesis math and return a typed intermediate result."""
        adjusted = self._adjust_markets(baseline=baseline, interpreted=interpreted)
        implied = self._market_implied_probabilities(odds)
        edge_market, edge_value = self._best_edge(adjusted=adjusted, implied=implied)
        confidence = self._confidence_score(interpreted)
        return SynthesisResult(
            adjusted=adjusted,
            confidence_score=confidence,
            edge_market=edge_market,
            edge_value=edge_value,
        )

    def _adjust_probability(self, baseline_p: float, signed_reliability_deltas: list[float]) -> float:
        """Shift baseline_p in log-odds space by the sum of signed reliability-weighted deltas, then sigmoid back."""
        clamped = min(max(baseline_p, self._tuning.probability_floor), self._tuning.probability_ceiling)
        log_odds = math.log(clamped / (1 - clamped))
        delta = sum(signed_reliability_deltas) * self._tuning.base_sensitivity
        adjusted = 1 / (1 + math.exp(-(log_odds + delta)))
        return round(adjusted, self._tuning.rounding_precision)

    def _implied_probability(self, odds: float) -> float:
        """Convert decimal odds to implied probability, without overround correction."""
        if odds <= 0:
            return self._tuning.implied_probability_default
        return round(1 / odds, self._tuning.rounding_precision)

    def _match_from_state(self, state: GraphState, match_id: str) -> Match | None:
        """Return the match object for the supplied match id from shared state."""
        return next((match for match in state.get("matches", []) if match.match_id == match_id), None)

    def _adjust_markets(self, baseline: BaselineForecast, interpreted: list[InterpretedEvidence]) -> BaselineForecast:
        """Apply evidence to the winner and goals markets, then renormalise each market independently."""
        raw_home = self._adjust_probability(
            baseline.home_win,
            [self.WINNER_SIGNS_HOME.get(e.winner_direction, 0.0) * e.reliability_score * e.market_weight for e in interpreted],
        )
        raw_away = self._adjust_probability(
            baseline.away_win,
            [self.WINNER_SIGNS_AWAY.get(e.winner_direction, 0.0) * e.reliability_score * e.market_weight for e in interpreted],
        )
        raw_draw = baseline.draw
        winner_total = raw_home + raw_draw + raw_away or 1.0

        raw_over = self._adjust_probability(
            baseline.over_2_5,
            [self.GOALS_SIGNS_OVER.get(e.goals_direction, 0.0) * e.reliability_score * e.market_weight for e in interpreted],
        )
        raw_under = self._adjust_probability(
            baseline.under_2_5,
            [self.GOALS_SIGNS_UNDER.get(e.goals_direction, 0.0) * e.reliability_score * e.market_weight for e in interpreted],
        )
        goals_total = raw_over + raw_under or 1.0

        return BaselineForecast(
            home_win=round(raw_home / winner_total, self._tuning.rounding_precision),
            draw=round(raw_draw / winner_total, self._tuning.rounding_precision),
            away_win=round(raw_away / winner_total, self._tuning.rounding_precision),
            over_2_5=round(raw_over / goals_total, self._tuning.rounding_precision),
            under_2_5=round(raw_under / goals_total, self._tuning.rounding_precision),
        )

    def _market_implied_probabilities(self, odds: MarketOdds) -> dict[str, float]:
        """Convert odds into overround-normalised implied probabilities for each supported market."""
        winner_probs = {
            "home_win": self._implied_probability(odds.home_win),
            "draw": self._implied_probability(odds.draw),
            "away_win": self._implied_probability(odds.away_win),
        }
        goals_probs = {
            "over_2_5": self._implied_probability(odds.over_2_5),
            "under_2_5": self._implied_probability(odds.under_2_5),
        }
        return {
            **self._normalise_market(winner_probs),
            **self._normalise_market(goals_probs),
        }

    def _normalise_market(self, probabilities: dict[str, float]) -> dict[str, float]:
        """Renormalise a market to sum to one after removing bookmaker overround."""
        total = sum(probabilities.values()) or 1.0
        return {market: round(value / total, self._tuning.rounding_precision) for market, value in probabilities.items()}

    def _best_edge(self, adjusted: BaselineForecast, implied: dict[str, float]) -> tuple[str | None, float | None]:
        """Return the market with the largest adjusted-minus-implied edge; the guard enforces a minimum threshold."""
        adjusted_map = {
            "home_win": adjusted.home_win,
            "draw": adjusted.draw,
            "away_win": adjusted.away_win,
            "over_2_5": adjusted.over_2_5,
            "under_2_5": adjusted.under_2_5,
        }
        edges = {
            market: round(
                probability - implied.get(market, self._tuning.implied_probability_default),
                self._tuning.rounding_precision,
            )
            for market, probability in adjusted_map.items()
        }
        best_market = max(edges, key=edges.get, default=None)
        if best_market is None:
            return None, None
        return best_market, edges[best_market]

    def _confidence_score(self, interpreted: list[InterpretedEvidence]) -> float:
        """Estimate confidence from evidence volume, reliability, and source diversity."""
        if not interpreted:
            return self._tuning.confidence_empty_evidence
        avg_reliability = sum(item.reliability_score for item in interpreted) / len(interpreted)
        source_bonus = min(
            len({item.source for item in interpreted}) * self._tuning.source_bonus_per_unique_source,
            self._tuning.source_bonus_cap,
        )
        count_bonus = min(
            len(interpreted) * self._tuning.count_bonus_per_item,
            self._tuning.count_bonus_cap,
        )
        confidence = (
            self._tuning.confidence_base
            + (avg_reliability * self._tuning.confidence_reliability_weight)
            + source_bonus
            + count_bonus
        )
        return round(min(self._tuning.confidence_cap, confidence), self._tuning.rounding_precision)

    def _build_rationale(
        self,
        match: Match,
        baseline: BaselineForecast,
        adjusted: BaselineForecast,
        interpreted: list[InterpretedEvidence],
        edge_market: str | None,
    ) -> str:
        """Ask the LLM for a short rationale, falling back to a deterministic summary if needed."""
        prompt = render_synthesis_rationale_messages(
            match=match,
            baseline=baseline,
            adjusted=adjusted,
            interpreted=interpreted,
            edge_market=edge_market,
            evidence_limit=self._tuning.rationale_evidence_limit,
            sentence_limit=self._tuning.rationale_sentence_limit,
        )
        try:
            response = self._llm.chat(
                messages=prompt,
                temperature=self._tuning.rationale_temperature,
                max_tokens=self._tuning.rationale_max_tokens,
            ).strip()
        except Exception:
            response = ""
        return response or self._fallback_rationale(
            match=match, interpreted=interpreted, edge_market=edge_market
        )

    def _fallback_rationale(
        self,
        match: Match,
        interpreted: list[InterpretedEvidence],
        edge_market: str | None,
    ) -> str:
        """Return a deterministic rationale when the LLM is unavailable."""
        strongest = max(interpreted, key=lambda item: item.reliability_score, default=None)
        if strongest is None:
            return (
                f"The baseline still drives the view for {match.home_team} vs {match.away_team}, "
                f"with the strongest edge in {edge_market or 'no market'}. "
                "Qualitative evidence was limited, so this should be treated cautiously."
            )
        return (
            f"The baseline leaned toward {match.home_team} vs {match.away_team} and the adjusted view remains most favorable "
            f"in {edge_market or 'the current market'}. Stronger qualitative support came from {strongest.source}, "
            f"which noted: {strongest.summary}"
        )

    def _append_guard_reasons(self, rationale: str, reasons: list[str]) -> str:
        """Append guardrail failure reasons to the rationale for auditability."""
        if not reasons:
            return rationale
        return f"{rationale}\n\nAlert withheld because: " + "; ".join(reasons)
