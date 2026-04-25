from datetime import datetime, timedelta

from soccer_forecast_agent.guardrails.alert_guard import AlertGuard, AlertGuardConfig
from soccer_forecast_agent.models.evidence import EvidenceItem


def test_alert_guard_passes_with_recent_diverse_reliable_evidence(sample_forecast, recent_evidence):
    guard = AlertGuard()

    result = guard.check(sample_forecast, recent_evidence)

    assert result.passed is True
    assert result.reasons == []


def test_alert_guard_collects_multiple_failure_reasons(sample_forecast):
    guard = AlertGuard(
        AlertGuardConfig(
            min_evidence_count=3,
            min_avg_reliability=0.6,
            spam_window_hours=6,
            min_odds_delta=0.05,
        )
    )
    weak_evidence = [
        EvidenceItem(
            evidence_id="weak-1",
            forecast_id="forecast-1",
            source="reddit.com",
            url="https://reddit.com/r/soccer",
            timestamp=datetime.utcnow() - timedelta(hours=72),
            summary="Unverified rumor about lineup uncertainty.",
            direction="uncertainty",
            reliability_score=0.2,
            applies_to_market="winner",
        )
    ]

    result = guard.check(
        sample_forecast,
        weak_evidence,
        prior_alert_at=datetime.utcnow() - timedelta(hours=1),
        odds_delta=0.01,
    )

    assert result.passed is False
    assert any("Insufficient evidence" in reason for reason in result.reasons)
    assert any("Low avg reliability" in reason for reason in result.reasons)
    assert any("All evidence older" in reason for reason in result.reasons)
    assert any("Insufficient source diversity" in reason for reason in result.reasons)
    assert any("Spam suppression" in reason for reason in result.reasons)


def test_alert_guard_handles_naive_datetimes_without_crashing(sample_forecast):
    guard = AlertGuard()
    evidence = [
        EvidenceItem(
            evidence_id="naive-1",
            forecast_id="forecast-1",
            source="bbc.co.uk",
            url="https://bbc.co.uk",
            timestamp=datetime.utcnow() - timedelta(hours=1),
            summary="Recent report with naive timestamp.",
            direction="home_positive",
            reliability_score=0.8,
            applies_to_market="winner",
        ),
        EvidenceItem(
            evidence_id="naive-2",
            forecast_id="forecast-1",
            source="skysports.com",
            url="https://skysports.com",
            timestamp=datetime.utcnow() - timedelta(hours=2),
            summary="Another recent report with naive timestamp.",
            direction="neutral",
            reliability_score=0.7,
            applies_to_market="both",
        ),
        EvidenceItem(
            evidence_id="naive-3",
            forecast_id="forecast-1",
            source="premierleague.com",
            url="https://premierleague.com",
            timestamp=datetime.utcnow() - timedelta(hours=3),
            summary="Official update with naive timestamp.",
            direction="home_positive",
            reliability_score=0.9,
            applies_to_market="winner",
        ),
    ]

    result = guard.check(sample_forecast, evidence)

    assert result.passed is True


def test_alert_guard_uses_default_config_values():
    guard = AlertGuard()

    assert guard._config == AlertGuardConfig()
