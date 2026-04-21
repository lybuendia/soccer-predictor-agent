"""Alert guardrail — decides whether a forecast meets the quality bar to trigger an alert."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from soccer_forecast_agent.models.evidence import EvidenceItem
from soccer_forecast_agent.models.match import Forecast


@dataclass
class GuardResult:
    """Outcome of a guardrail check with a pass/fail flag and list of failure reasons."""

    passed: bool
    reasons: list[str] = field(default_factory=list)


class AlertGuard:
    """Enforces evidence quality, confidence, edge, and spam suppression rules before any alert fires."""

    def __init__(
        self,
        min_evidence_count: int = 3,
        min_avg_reliability: float = 0.5,
        max_evidence_age_hours: int = 48,
        min_unique_sources: int = 2,
        min_edge_threshold: float = 0.05,
        min_confidence_threshold: float = 0.60,
        spam_window_hours: int = 6,
        min_odds_delta: float = 0.05,
    ) -> None:
        """Initialise with configurable thresholds for all guard checks."""
        self._min_evidence_count = min_evidence_count
        self._min_avg_reliability = min_avg_reliability
        self._max_evidence_age_hours = max_evidence_age_hours
        self._min_unique_sources = min_unique_sources
        self._min_edge_threshold = min_edge_threshold
        self._min_confidence_threshold = min_confidence_threshold
        self._spam_window_hours = spam_window_hours
        self._min_odds_delta = min_odds_delta

    def check(
        self,
        forecast: Forecast,
        evidence: list[EvidenceItem],
        prior_alert_at: datetime | None = None,
        odds_delta: float = 0.0,
    ) -> GuardResult:
        """Run all guards and return a GuardResult. All checks run; reasons accumulate."""
        reasons: list[str] = []

        if len(evidence) < self._min_evidence_count:
            reasons.append(f"Insufficient evidence: {len(evidence)} < {self._min_evidence_count}")

        if evidence:
            avg_reliability = sum(e.reliability_score for e in evidence) / len(evidence)
            if avg_reliability < self._min_avg_reliability:
                reasons.append(f"Low avg reliability: {avg_reliability:.2f} < {self._min_avg_reliability}")

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self._max_evidence_age_hours)
        stale = [e for e in evidence if self._as_utc(e.timestamp) < cutoff]
        if len(stale) == len(evidence):
            reasons.append(f"All evidence older than {self._max_evidence_age_hours}h")

        unique_sources = len({e.source for e in evidence})
        if unique_sources < self._min_unique_sources:
            reasons.append(f"Insufficient source diversity: {unique_sources} < {self._min_unique_sources}")

        if forecast.edge_value is None or forecast.edge_value < self._min_edge_threshold:
            reasons.append(f"Edge too small: {forecast.edge_value} < {self._min_edge_threshold}")

        if forecast.confidence_score < self._min_confidence_threshold:
            reasons.append(f"Confidence too low: {forecast.confidence_score:.2f} < {self._min_confidence_threshold}")

        if prior_alert_at is not None:
            window = timedelta(hours=self._spam_window_hours)
            if now - self._as_utc(prior_alert_at) < window and odds_delta < self._min_odds_delta:
                reasons.append(f"Spam suppression: alerted within {self._spam_window_hours}h with no material odds change")

        return GuardResult(passed=len(reasons) == 0, reasons=reasons)

    def _as_utc(self, value: datetime) -> datetime:
        """Return a timezone-aware UTC datetime, treating naive values as UTC."""
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
