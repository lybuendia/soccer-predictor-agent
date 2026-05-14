from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from soccer_forecast_agent.agents.synthesis_alert import SynthesisAlertAgent, SynthesisTuning
from soccer_forecast_agent.guardrails.alert_guard import AlertGuard, AlertGuardConfig
from soccer_forecast_agent.models.evidence import EvidenceItem, InterpretedEvidence
from soccer_forecast_agent.models.match import AlertPayload, BaselineForecast, Forecast, SynthesisDecision


class FakeLLM:
    def __init__(self, response: str | None = None) -> None:
        if response is None:
            response = (
                '{"market_category":"winner","recommended_market":"home_win",'
                '"winner_adjustment":"strong_home","goals_adjustment":"neutral",'
                '"llm_conviction_score":1.0,"alert_worthy":true,'
                '"summary":"Home side should improve on the baseline.",'
                '"rationale_points":["The home squad news is supportive."],'
                '"risk_points":["The edge still depends on limited qualitative evidence."]}'
            )
        self.response = response
        self.calls: list[dict] = []

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.response

    def chat_with_tools(self, messages: list[dict[str, str]], tools: list[dict], **kwargs) -> dict:
        raise AssertionError("chat_with_tools should not be used by SynthesisAlertAgent")


@dataclass
class FakeAlertChannel:
    should_succeed: bool = True

    def __post_init__(self) -> None:
        self.payloads: list[AlertPayload] = []

    def send(self, payload: AlertPayload) -> bool:
        self.payloads.append(payload)
        return self.should_succeed


class FakeForecastRepository:
    def __init__(self, latest: Forecast | None = None) -> None:
        self.latest = latest
        self.saved: list[Forecast] = []

    def save_forecast(self, forecast: Forecast) -> None:
        self.saved.append(forecast)
        self.latest = forecast

    def get_by_match(self, match_id: str) -> list[Forecast]:
        return [forecast for forecast in self.saved if forecast.match_id == match_id]

    def get_latest(self, match_id: str) -> Forecast | None:
        if self.latest and self.latest.match_id == match_id:
            return self.latest
        return None


class FakeEvidenceRepository:
    def __init__(self) -> None:
        self.saved: list[EvidenceItem] = []

    def save_evidence(self, item: EvidenceItem) -> None:
        self.saved.append(item)


def _state(
    sample_match,
    sample_baseline,
    sample_odds,
    evidence_items: list[EvidenceItem],
    interpreted_evidence: list[InterpretedEvidence],
) -> dict:
    return {
        "competition": "PL",
        "days_ahead": 7,
        "matches": [sample_match],
        "odds_map": {sample_match.match_id: sample_odds},
        "baseline_forecasts": {sample_match.match_id: sample_baseline},
        "current_match_id": sample_match.match_id,
        "current_forecast_id": "forecast-123",
        "evidence_items": evidence_items,
        "interpreted_evidence": interpreted_evidence,
        "forecast": None,
        "all_forecasts": [],
        "alert_sent": False,
        "errors": [],
    }


def test_synthesis_alert_agent_sends_alert_when_guard_passes(
    sample_match, sample_baseline, sample_odds, recent_evidence, recent_interpreted_evidence
) -> None:
    llm = FakeLLM()
    alert_channel = FakeAlertChannel(should_succeed=True)
    forecast_repo = FakeForecastRepository()
    evidence_repo = FakeEvidenceRepository()
    guard = AlertGuard(AlertGuardConfig(min_edge_threshold=0.01, min_confidence_threshold=0.6))
    agent = SynthesisAlertAgent(llm, alert_channel, forecast_repo, evidence_repo, guard, tuning=SynthesisTuning())

    result = agent.run(_state(sample_match, sample_baseline, sample_odds, recent_evidence, recent_interpreted_evidence))

    assert result["alert_sent"] is True
    assert result["forecast"] is not None
    assert result["forecast"].alert_sent is True
    assert result["forecast"].edge_market is not None
    assert result["forecast"].edge_value is not None
    assert len(forecast_repo.saved) == 1
    assert [item.forecast_id for item in evidence_repo.saved] == ["forecast-123", "forecast-123", "forecast-123"]
    assert len(alert_channel.payloads) == 1
    assert alert_channel.payloads[0].market_odds is sample_odds
    assert llm.calls
    assert llm.calls[0]["kwargs"]["system"]
    assert all(message["role"] != "system" for message in llm.calls[0]["messages"])


def test_synthesis_alert_agent_withholds_alert_and_records_reasons_when_guard_fails(
    sample_match,
    sample_baseline,
    sample_odds,
) -> None:
    weak_evidence = [
        EvidenceItem(
            evidence_id="weak-1",
            forecast_id="forecast-1",
            source="rumorblog.example",
            url="https://rumorblog.example/story",
            timestamp=datetime.now(timezone.utc),
            summary="Unconfirmed report about a possible rotation.",
            direction="uncertainty",
            reliability_score=0.2,
            applies_to_market="winner",
        )
    ]
    weak_interpreted = [
        InterpretedEvidence(
            evidence_id="weak-1",
            source="rumorblog.example",
            reliability_score=0.2,
            winner_direction="uncertainty",
            goals_direction="neutral",
            market_weight=0.7,
            summary="Unconfirmed report about a possible rotation.",
        )
    ]
    llm = FakeLLM(
        '{"market_category":"winner","recommended_market":"away_win",'
        '"winner_adjustment":"light_away","goals_adjustment":"neutral",'
        '"llm_conviction_score":0.3,"alert_worthy":false,'
        '"summary":"Evidence is too weak to justify an alert.",'
        '"rationale_points":["Only one low-reliability report was available."],'
        '"risk_points":["The signal may be noise."]}'
    )
    alert_channel = FakeAlertChannel(should_succeed=True)
    forecast_repo = FakeForecastRepository()
    evidence_repo = FakeEvidenceRepository()
    guard = AlertGuard(
        AlertGuardConfig(
            min_evidence_count=3,
            min_avg_reliability=0.6,
            min_edge_threshold=0.05,
            min_confidence_threshold=0.7,
        )
    )
    agent = SynthesisAlertAgent(llm, alert_channel, forecast_repo, evidence_repo, guard, tuning=SynthesisTuning())

    result = agent.run(_state(sample_match, sample_baseline, sample_odds, weak_evidence, weak_interpreted))

    assert result["alert_sent"] is False
    assert result["forecast"] is not None
    assert result["forecast"].alert_sent is False
    assert "Alert withheld because:" in result["forecast"].rationale
    assert "Evidence is too weak to justify an alert." in result["forecast"].rationale
    assert len(alert_channel.payloads) == 0
    assert len(forecast_repo.saved) == 1


def _make_agent() -> SynthesisAlertAgent:
    return SynthesisAlertAgent(
        llm=FakeLLM(),
        alert_channel=FakeAlertChannel(),
        forecast_repo=FakeForecastRepository(),
        evidence_repo=FakeEvidenceRepository(),
        guard=AlertGuard(),
        tuning=SynthesisTuning(),
    )


def _baseline() -> BaselineForecast:
    return BaselineForecast(home_win=0.45, draw=0.26, away_win=0.29, over_2_5=0.55, under_2_5=0.45)


def test_draw_positive_evidence_increases_draw_share() -> None:
    agent = _make_agent()
    baseline = _baseline()

    decision = SynthesisDecision(
        market_category="winner",
        recommended_market="draw",
        winner_adjustment="medium_draw",
        goals_adjustment="neutral",
        llm_conviction_score=0.8,
        alert_worthy=True,
    )
    adjusted = agent._adjust_markets(baseline=baseline, decision=decision, evidence_quality_score=0.8)
    assert adjusted.draw > baseline.draw, "draw_positive evidence should increase the draw share after renormalisation"
    assert adjusted.home_win < baseline.home_win
    assert adjusted.away_win < baseline.away_win
    assert abs(adjusted.home_win + adjusted.draw + adjusted.away_win - 1.0) < 1e-4


def test_single_market_evidence_is_discounted_vs_both_market_evidence() -> None:
    agent = _make_agent()
    baseline = _baseline()

    light = SynthesisDecision(
        market_category="winner",
        recommended_market="home_win",
        winner_adjustment="light_home",
        goals_adjustment="neutral",
        llm_conviction_score=0.8,
        alert_worthy=True,
    )
    medium = SynthesisDecision(
        market_category="winner",
        recommended_market="home_win",
        winner_adjustment="medium_home",
        goals_adjustment="neutral",
        llm_conviction_score=0.8,
        alert_worthy=True,
    )
    adj_light = agent._adjust_markets(baseline=baseline, decision=light, evidence_quality_score=0.8)
    adj_medium = agent._adjust_markets(baseline=baseline, decision=medium, evidence_quality_score=0.8)
    assert adj_medium.home_win > adj_light.home_win


def test_llm_conviction_influences_confidence_score() -> None:
    agent = _make_agent()
    low = agent._confidence_score(0.8, 0.2)
    high = agent._confidence_score(0.8, 0.9)
    assert high > low
