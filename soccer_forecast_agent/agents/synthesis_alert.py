"""Synthesis and Alert agent — combines baseline, evidence, and LLM adjudication into a final alert decision."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from soccer_forecast_agent.agents.supervisor import GraphState
from soccer_forecast_agent.guardrails.alert_guard import AlertGuard, GuardResult
from soccer_forecast_agent.memory.repository import EvidenceRepository, ForecastRepository
from soccer_forecast_agent.models.evidence import EvidenceItem, InterpretedEvidence
from soccer_forecast_agent.models.match import (
    AlertPayload,
    BaselineForecast,
    Forecast,
    Match,
    MarketOdds,
    SynthesisDecision,
    SynthesisResult,
)
from soccer_forecast_agent.prompts.loader import render_synthesis_decision_messages
from soccer_forecast_agent.providers.llm import LLMProvider
from soccer_forecast_agent.tools.alert_channel import AlertChannel


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthesisTuning:
    """Typed tuning values for bounded synthesis adjustments and confidence scoring."""

    base_sensitivity: float = 0.15
    probability_floor: float = 0.01
    probability_ceiling: float = 0.99
    implied_probability_default: float = 0.0
    prior_odds_delta_default: float = 0.0
    evidence_quality_cap: float = 1.0
    evidence_quality_empty: float = 0.0
    source_bonus_per_unique_source: float = 0.05
    source_bonus_cap: float = 0.15
    count_bonus_per_item: float = 0.03
    count_bonus_cap: float = 0.15
    confidence_base: float = 0.15
    confidence_quality_weight: float = 0.45
    confidence_llm_weight: float = 0.35
    confidence_cap: float = 0.95
    rounding_precision: int = 4
    decision_evidence_limit: int = 6
    decision_temperature: float = 0.0
    decision_max_tokens: int = 350


class SynthesisAlertAgent:
    """Uses a bounded LLM synthesis decision to move the baseline and decide alert-worthiness."""

    WINNER_ADJUSTMENTS = {
        "strong_home": (1.0, -1.0),
        "medium_home": (0.7, -0.7),
        "light_home": (0.35, -0.35),
        "neutral": (0.0, 0.0),
        "light_draw": (-0.2, -0.2),
        "medium_draw": (-0.4, -0.4),
        "light_away": (-0.35, 0.35),
        "medium_away": (-0.7, 0.7),
        "strong_away": (-1.0, 1.0),
    }
    GOALS_ADJUSTMENTS = {
        "strong_over": (1.0, -1.0),
        "medium_over": (0.7, -0.7),
        "light_over": (0.35, -0.35),
        "neutral": (0.0, 0.0),
        "light_under": (-0.35, 0.35),
        "medium_under": (-0.7, 0.7),
        "strong_under": (-1.0, 1.0),
    }
    VALID_MARKET_CATEGORIES = {"winner", "goals"}
    VALID_RECOMMENDED_MARKETS = {"home_win", "draw", "away_win", "over_2_5", "under_2_5"}

    def __init__(
        self,
        llm: LLMProvider,
        alert_channel: AlertChannel,
        forecast_repo: ForecastRepository,
        evidence_repo: EvidenceRepository,
        guard: AlertGuard,
        tuning: SynthesisTuning | None = None,
    ) -> None:
        """Initialise with injected LLM, alert channel, persistence, and guardrail."""
        self._llm = llm
        self._alert = alert_channel
        self._forecast_repo = forecast_repo
        self._evidence_repo = evidence_repo
        self._guard = guard
        self._tuning = tuning or SynthesisTuning()

    def run(self, state: GraphState) -> GraphState:
        """Apply LLM-guided bounded adjustment, evaluate the guard, send alert if thresholds pass, and persist forecast."""
        current_match_id = state.get("current_match_id")
        current_forecast_id = state.get("current_forecast_id")
        errors = list(state.get("errors", []))
        if not current_match_id:
            errors.append("SynthesisAlertAgent requires current_match_id in state")
            return {**state, "errors": errors}
        if not current_forecast_id:
            errors.append("SynthesisAlertAgent requires current_forecast_id in state")
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
        evidence_quality = self._evidence_quality_score(interpreted)
        decision = self._request_synthesis_decision(
            match=match,
            baseline=baseline,
            odds=odds,
            interpreted=interpreted,
            evidence_quality_score=evidence_quality,
        )
        synthesis = self._synthesise(
            baseline=baseline,
            decision=decision,
            odds=odds,
            evidence_quality_score=evidence_quality,
        )
        LOGGER.debug(
            "SynthesisAlertAgent decision for match_id=%s: %s",
            current_match_id,
            self._pretty(dataclasses.asdict(decision)),
        )
        LOGGER.debug(
            "SynthesisAlertAgent synthesis for match_id=%s: %s",
            current_match_id,
            self._pretty(dataclasses.asdict(synthesis)),
        )

        rationale = self._build_rationale(match=match, decision=decision, synthesis=synthesis)
        preliminary = Forecast(
            forecast_id=current_forecast_id,
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
        guard_result = self._merge_llm_decision_guard(guard_result, decision)
        final_rationale = (
            self._append_guard_reasons(rationale, guard_result.reasons)
            if not guard_result.passed
            else rationale
        )

        forecast = dataclasses.replace(preliminary, rationale=final_rationale)
        alert_sent = False
        if guard_result.passed:
            alert_sent = self._alert.send(
                AlertPayload(
                    match=match,
                    forecast=forecast,
                    market_odds=odds,
                    rationale=final_rationale,
                )
            )
            if not alert_sent:
                errors.append(f"Alert delivery failed for match_id={current_match_id}")

        forecast = dataclasses.replace(forecast, alert_sent=alert_sent)
        LOGGER.info(
            "Forecast summary for %s vs %s: %s",
            match.home_team,
            match.away_team,
            self._human_summary(
                match=match,
                forecast=forecast,
                odds=odds,
                decision=decision,
                guard_reasons=guard_result.reasons,
            ),
        )
        LOGGER.debug(
            "SynthesisAlertAgent final forecast for match_id=%s: %s",
            current_match_id,
            self._pretty(self._forecast_snapshot(forecast)),
        )

        self._forecast_repo.save_forecast(forecast)
        errors.extend(self._persist_evidence_for_forecast(forecast.forecast_id, evidence))
        all_forecasts = list(state.get("all_forecasts", [])) + [forecast]
        return {
            **state,
            "forecast": forecast,
            "all_forecasts": all_forecasts,
            "alert_sent": alert_sent,
            "errors": errors,
        }

    def _request_synthesis_decision(
        self,
        match: Match,
        baseline: BaselineForecast,
        odds: MarketOdds,
        interpreted: list[InterpretedEvidence],
        evidence_quality_score: float,
    ) -> SynthesisDecision:
        """Ask the synthesis LLM for a structured adjudication over the baseline and evidence."""
        prompt = render_synthesis_decision_messages(
            match=match,
            baseline=baseline,
            odds=odds,
            interpreted=interpreted,
            evidence_quality_score=evidence_quality_score,
            evidence_limit=self._tuning.decision_evidence_limit,
        )
        try:
            response = self._llm.chat(
                messages=self._chat_messages_without_system(prompt),
                system=self._chat_system_prompt(prompt),
                temperature=self._tuning.decision_temperature,
                max_tokens=self._tuning.decision_max_tokens,
            )
            LOGGER.debug("SynthesisAlertAgent decision raw response: %s", response)
            parsed = self._parse_json_object(response)
            return self._normalise_decision(parsed)
        except Exception as exc:
            LOGGER.warning("Synthesis decision LLM call failed; using deterministic fallback. Error: %s", exc)
            return self._fallback_decision(interpreted=interpreted, evidence_quality_score=evidence_quality_score)

    def _normalise_decision(self, raw: dict) -> SynthesisDecision:
        """Validate and normalise the LLM's structured synthesis decision."""
        market_category = str(raw.get("market_category", "winner")).strip()
        if market_category not in self.VALID_MARKET_CATEGORIES:
            market_category = "winner"
        recommended_market = str(raw.get("recommended_market", "home_win")).strip()
        if recommended_market not in self.VALID_RECOMMENDED_MARKETS:
            recommended_market = "home_win"
        winner_adjustment = str(raw.get("winner_adjustment", "neutral")).strip()
        if winner_adjustment not in self.WINNER_ADJUSTMENTS:
            winner_adjustment = "neutral"
        goals_adjustment = str(raw.get("goals_adjustment", "neutral")).strip()
        if goals_adjustment not in self.GOALS_ADJUSTMENTS:
            goals_adjustment = "neutral"
        conviction = self._clamp_float(raw.get("llm_conviction_score", 0.5))
        alert_worthy = bool(raw.get("alert_worthy", False))
        rationale_points = self._normalise_list(raw.get("rationale_points"))
        risk_points = self._normalise_list(raw.get("risk_points"))
        summary = str(raw.get("summary", "")).strip()
        return SynthesisDecision(
            market_category=market_category,
            recommended_market=recommended_market,
            winner_adjustment=winner_adjustment,
            goals_adjustment=goals_adjustment,
            llm_conviction_score=conviction,
            alert_worthy=alert_worthy,
            rationale_points=rationale_points,
            risk_points=risk_points,
            summary=summary,
        )

    def _fallback_decision(self, interpreted: list[InterpretedEvidence], evidence_quality_score: float) -> SynthesisDecision:
        """Build a deterministic synthesis decision when the LLM is unavailable."""
        winner_score = sum(
            self._winner_signal(item.winner_direction) * item.reliability_score * item.market_weight
            for item in interpreted
        )
        goals_score = sum(
            self._goals_signal(item.goals_direction) * item.reliability_score * item.market_weight
            for item in interpreted
        )
        winner_adjustment = self._winner_adjustment_from_score(winner_score)
        goals_adjustment = self._goals_adjustment_from_score(goals_score)
        recommended_market = self._fallback_market_from_scores(winner_score, goals_score)
        market_category = "goals" if recommended_market in {"over_2_5", "under_2_5"} else "winner"
        alert_worthy = evidence_quality_score >= 0.55 and recommended_market != "draw"
        summary = "Fallback synthesis decision based on aggregated evidence directions."
        rationale_points = ["Evidence directions were aggregated deterministically because the synthesis LLM was unavailable."]
        risk_points = ["The synthesis recommendation may be less nuanced than the normal LLM-guided path."]
        return SynthesisDecision(
            market_category=market_category,
            recommended_market=recommended_market,
            winner_adjustment=winner_adjustment,
            goals_adjustment=goals_adjustment,
            llm_conviction_score=evidence_quality_score,
            alert_worthy=alert_worthy,
            rationale_points=rationale_points,
            risk_points=risk_points,
            summary=summary,
        )

    def _synthesise(
        self,
        baseline: BaselineForecast,
        decision: SynthesisDecision,
        odds: MarketOdds,
        evidence_quality_score: float,
    ) -> SynthesisResult:
        """Apply the bounded LLM synthesis decision and return the adjusted forecast summary."""
        adjusted = self._adjust_markets(
            baseline=baseline,
            decision=decision,
            evidence_quality_score=evidence_quality_score,
        )
        implied = self._market_implied_probabilities(odds)
        adjusted_probability = self._probability_from_baseline(adjusted, decision.recommended_market)
        edge_market = decision.recommended_market
        edge_value = (
            round(
                adjusted_probability - implied.get(edge_market, self._tuning.implied_probability_default),
                self._tuning.rounding_precision,
            )
            if adjusted_probability is not None
            else None
        )
        confidence = self._confidence_score(evidence_quality_score, decision.llm_conviction_score)
        return SynthesisResult(
            adjusted=adjusted,
            confidence_score=confidence,
            edge_market=edge_market,
            edge_value=edge_value,
        )

    def _adjust_markets(
        self,
        baseline: BaselineForecast,
        decision: SynthesisDecision,
        evidence_quality_score: float,
    ) -> BaselineForecast:
        """Apply the bounded synthesis decision to the baseline and renormalise each market."""
        quality_scale = self._clamp_float(evidence_quality_score)
        conviction_scale = self._clamp_float(decision.llm_conviction_score)
        scale = quality_scale * conviction_scale

        winner_home_delta, winner_away_delta = self.WINNER_ADJUSTMENTS[decision.winner_adjustment]
        raw_home = self._adjust_probability(baseline.home_win, [winner_home_delta * scale])
        raw_away = self._adjust_probability(baseline.away_win, [winner_away_delta * scale])
        raw_draw = baseline.draw
        winner_total = raw_home + raw_draw + raw_away or 1.0

        goals_over_delta, goals_under_delta = self.GOALS_ADJUSTMENTS[decision.goals_adjustment]
        raw_over = self._adjust_probability(baseline.over_2_5, [goals_over_delta * scale])
        raw_under = self._adjust_probability(baseline.under_2_5, [goals_under_delta * scale])
        goals_total = raw_over + raw_under or 1.0

        return BaselineForecast(
            home_win=round(raw_home / winner_total, self._tuning.rounding_precision),
            draw=round(raw_draw / winner_total, self._tuning.rounding_precision),
            away_win=round(raw_away / winner_total, self._tuning.rounding_precision),
            over_2_5=round(raw_over / goals_total, self._tuning.rounding_precision),
            under_2_5=round(raw_under / goals_total, self._tuning.rounding_precision),
        )

    def _adjust_probability(self, baseline_p: float, signed_deltas: list[float]) -> float:
        """Shift a probability in log-odds space using bounded, code-owned deltas."""
        clamped = min(max(baseline_p, self._tuning.probability_floor), self._tuning.probability_ceiling)
        log_odds = math.log(clamped / (1 - clamped))
        adjusted = 1 / (1 + math.exp(-(log_odds + (sum(signed_deltas) * self._tuning.base_sensitivity))))
        return round(adjusted, self._tuning.rounding_precision)

    def _evidence_quality_score(self, interpreted: list[InterpretedEvidence]) -> float:
        """Estimate deterministic evidence quality from reliability, source diversity, and count."""
        if not interpreted:
            return self._tuning.evidence_quality_empty
        avg_reliability = sum(item.reliability_score for item in interpreted) / len(interpreted)
        source_bonus = min(
            len({item.source for item in interpreted}) * self._tuning.source_bonus_per_unique_source,
            self._tuning.source_bonus_cap,
        )
        count_bonus = min(
            len(interpreted) * self._tuning.count_bonus_per_item,
            self._tuning.count_bonus_cap,
        )
        score = min(self._tuning.evidence_quality_cap, avg_reliability + source_bonus + count_bonus)
        return round(score, self._tuning.rounding_precision)

    def _confidence_score(self, evidence_quality_score: float, llm_conviction_score: float) -> float:
        """Blend deterministic evidence quality with the synthesis LLM's conviction."""
        confidence = (
            self._tuning.confidence_base
            + (evidence_quality_score * self._tuning.confidence_quality_weight)
            + (llm_conviction_score * self._tuning.confidence_llm_weight)
        )
        return round(min(self._tuning.confidence_cap, confidence), self._tuning.rounding_precision)

    def _merge_llm_decision_guard(self, guard_result: GuardResult, decision: SynthesisDecision) -> GuardResult:
        """Combine hard guardrails with the synthesis LLM's alert-worthiness judgment."""
        reasons = list(guard_result.reasons)
        if not decision.alert_worthy:
            reasons.append("Synthesis LLM judged the signal not alert-worthy before hard guardrails.")
        return GuardResult(passed=guard_result.passed and decision.alert_worthy, reasons=reasons)

    def _build_rationale(self, match: Match, decision: SynthesisDecision, synthesis: SynthesisResult) -> str:
        """Build a user-facing rationale directly from the synthesis decision."""
        lines: list[str] = []
        if decision.summary:
            lines.append(decision.summary.strip())
        if decision.rationale_points:
            lines.append("Key reasons: " + "; ".join(decision.rationale_points[:3]))
        if decision.risk_points:
            lines.append("Main risks: " + "; ".join(decision.risk_points[:2]))
        if not lines:
            lines.append(
                f"The baseline for {match.home_team} vs {match.away_team} was moved conservatively by the synthesis adjudication."
            )
        lines.append(
            f"Recommended market: {decision.recommended_market}; adjusted edge: {(synthesis.edge_value or 0.0):+.1%}."
        )
        return " ".join(lines)

    def _persist_evidence_for_forecast(self, forecast_id: str, evidence: list[EvidenceItem]) -> list[str]:
        """Persist evidence after the forecast exists so the foreign-key link stays correct."""
        errors: list[str] = []
        for item in evidence:
            evidence_item = dataclasses.replace(item, forecast_id=forecast_id)
            try:
                self._evidence_repo.save_evidence(evidence_item)
            except Exception as exc:
                errors.append(f"Failed to save evidence {evidence_item.evidence_id}: {exc}")
        return errors

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
        return {**self._normalise_market(winner_probs), **self._normalise_market(goals_probs)}

    def _normalise_market(self, probabilities: dict[str, float]) -> dict[str, float]:
        """Renormalise a market to sum to one after removing bookmaker overround."""
        total = sum(probabilities.values()) or 1.0
        return {
            market: round(value / total, self._tuning.rounding_precision)
            for market, value in probabilities.items()
        }

    def _implied_probability(self, odds: float) -> float:
        """Convert decimal odds to implied probability, without overround correction."""
        if odds <= 0:
            return self._tuning.implied_probability_default
        return round(1 / odds, self._tuning.rounding_precision)

    def _match_from_state(self, state: GraphState, match_id: str) -> Match | None:
        """Return the match object for the supplied match id from shared state."""
        return next((match for match in state.get("matches", []) if match.match_id == match_id), None)

    def _forecast_snapshot(self, forecast: Forecast) -> dict[str, object]:
        """Return a condensed forecast shape for debug logging."""
        return {
            "forecast_id": forecast.forecast_id,
            "match_id": forecast.match_id,
            "adjusted_home_win": forecast.adjusted_home_win,
            "adjusted_draw": forecast.adjusted_draw,
            "adjusted_away_win": forecast.adjusted_away_win,
            "adjusted_over_2_5": forecast.adjusted_over_2_5,
            "adjusted_under_2_5": forecast.adjusted_under_2_5,
            "confidence_score": forecast.confidence_score,
            "edge_market": forecast.edge_market,
            "edge_value": forecast.edge_value,
            "alert_sent": forecast.alert_sent,
            "rationale": forecast.rationale,
        }

    def _human_summary(
        self,
        match: Match,
        forecast: Forecast,
        odds: MarketOdds,
        decision: SynthesisDecision,
        guard_reasons: list[str],
    ) -> str:
        """Return a short human-readable summary of the forecast and alert decision."""
        market_category = self._market_category(decision.recommended_market)
        market_label = self._market_label(match, decision.recommended_market)
        adjusted_probability = self._probability_from_forecast(forecast, decision.recommended_market)
        edge_value = forecast.edge_value or 0.0
        implied_probability = (
            adjusted_probability - edge_value
            if adjusted_probability is not None and forecast.edge_value is not None
            else None
        )
        source_summary = self._market_source_summary(odds, decision.recommended_market)
        status = "Alert sent." if forecast.alert_sent else "Alert withheld."
        reason_suffix = f" Reason: {'; '.join(guard_reasons)}." if guard_reasons else ""
        probability_summary = (
            f"Our estimate is {adjusted_probability:.1%} versus a market-implied {implied_probability:.1%}, "
            f"for an edge of {edge_value:+.1%}."
            if adjusted_probability is not None and implied_probability is not None
            else "Adjusted and market-implied probabilities were unavailable."
        )
        return (
            f"Market category: {market_category}. Recommended bet: {market_label}. {probability_summary} "
            f"LLM conviction: {decision.llm_conviction_score:.1%}. Confidence: {forecast.confidence_score:.1%}. "
            f"Market source: {source_summary}. {status}{reason_suffix}"
        )

    def _market_category(self, market: str | None) -> str:
        """Return the broad market category for a market id."""
        if market in {"home_win", "draw", "away_win"}:
            return "Match winner"
        if market in {"over_2_5", "under_2_5"}:
            return "Goals total"
        return "Unknown market"

    def _market_label(self, match: Match, market: str | None) -> str:
        """Return a human-readable label for a market id."""
        labels = {
            "home_win": f"{match.home_team} win",
            "draw": "Draw",
            "away_win": f"{match.away_team} win",
            "over_2_5": "Over 2.5 goals",
            "under_2_5": "Under 2.5 goals",
        }
        return labels.get(market or "", market or "Unknown market")

    def _probability_from_baseline(self, baseline: BaselineForecast, market: str | None) -> float | None:
        """Return the probability for the given market from an adjusted BaselineForecast."""
        return {
            "home_win": baseline.home_win,
            "draw": baseline.draw,
            "away_win": baseline.away_win,
            "over_2_5": baseline.over_2_5,
            "under_2_5": baseline.under_2_5,
        }.get(market or "")

    def _probability_from_forecast(self, forecast: Forecast, market: str | None) -> float | None:
        """Return the adjusted probability for the given market from a persisted Forecast."""
        return {
            "home_win": forecast.adjusted_home_win,
            "draw": forecast.adjusted_draw,
            "away_win": forecast.adjusted_away_win,
            "over_2_5": forecast.adjusted_over_2_5,
            "under_2_5": forecast.adjusted_under_2_5,
        }.get(market or "")

    def _market_source_summary(self, odds: MarketOdds, market: str | None) -> str:
        """Return the market source summary for the selected market."""
        primary_source = odds.goals_market_source if market in {"over_2_5", "under_2_5"} else odds.winner_market_source
        seen_suffix = f" Other books seen: {', '.join(odds.market_sources_seen[:4])}." if odds.market_sources_seen else ""
        return f"The Odds API via {primary_source}.{seen_suffix}"

    def _parse_json_object(self, raw_text: str) -> dict:
        """Parse a JSON object from a model response, tolerating fenced code blocks."""
        candidate = raw_text.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[1]
            if candidate.endswith("```"):
                candidate = candidate.rsplit("\n", 1)[0]
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in synthesis decision response.")
        parsed = json.loads(candidate[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Synthesis decision response was not a JSON object.")
        return parsed

    def _normalise_list(self, value: object) -> list[str]:
        """Normalise an arbitrary value into a clean list of short strings."""
        if not isinstance(value, list):
            return []
        return [str(item).strip()[:200] for item in value if str(item).strip()]

    def _clamp_float(self, value: object) -> float:
        """Clamp a float-like value into the closed interval [0, 1]."""
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _winner_signal(self, direction: str) -> float:
        """Map winner-direction labels onto signed aggregate scores."""
        return {
            "home_positive": 1.0,
            "away_positive": -1.0,
            "draw_positive": 0.2,
            "neutral": 0.0,
            "uncertainty": 0.0,
        }.get(direction, 0.0)

    def _goals_signal(self, direction: str) -> float:
        """Map goals-direction labels onto signed aggregate scores."""
        return {
            "over_positive": 1.0,
            "under_positive": -1.0,
            "neutral": 0.0,
            "uncertainty": 0.0,
        }.get(direction, 0.0)

    def _winner_adjustment_from_score(self, score: float) -> str:
        """Convert an aggregate winner score into a bounded adjustment label."""
        magnitude = abs(score)
        if score >= 1.1:
            return "strong_home"
        if score >= 0.6:
            return "medium_home"
        if score >= 0.2:
            return "light_home"
        if score <= -1.1:
            return "strong_away"
        if score <= -0.6:
            return "medium_away"
        if score <= -0.2:
            return "light_away"
        if magnitude < 0.2:
            return "neutral"
        return "medium_draw"

    def _goals_adjustment_from_score(self, score: float) -> str:
        """Convert an aggregate goals score into a bounded adjustment label."""
        if score >= 1.1:
            return "strong_over"
        if score >= 0.6:
            return "medium_over"
        if score >= 0.2:
            return "light_over"
        if score <= -1.1:
            return "strong_under"
        if score <= -0.6:
            return "medium_under"
        if score <= -0.2:
            return "light_under"
        return "neutral"

    def _fallback_market_from_scores(self, winner_score: float, goals_score: float) -> str:
        """Choose a fallback market from aggregate winner/goals evidence scores."""
        if abs(goals_score) > abs(winner_score):
            return "over_2_5" if goals_score >= 0 else "under_2_5"
        if winner_score >= 0.2:
            return "home_win"
        if winner_score <= -0.2:
            return "away_win"
        return "draw"

    def _append_guard_reasons(self, rationale: str, reasons: list[str]) -> str:
        """Append guardrail failure reasons to the rationale for auditability."""
        if not reasons:
            return rationale
        return f"{rationale}\n\nAlert withheld because: " + "; ".join(reasons)

    def _chat_system_prompt(self, messages: list[dict[str, str]]) -> str | None:
        """Extract the leading system prompt so providers that require a top-level field can use it."""
        if messages and messages[0].get("role") == "system":
            return messages[0].get("content", "")
        return None

    def _chat_messages_without_system(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Return the chat history without the leading system message when one is present."""
        if messages and messages[0].get("role") == "system":
            return messages[1:]
        return messages

    def _pretty(self, value: object) -> str:
        """Return a compact JSON string for debug logging."""
        try:
            return json.dumps(value, indent=2, sort_keys=True)
        except TypeError:
            return str(value)
