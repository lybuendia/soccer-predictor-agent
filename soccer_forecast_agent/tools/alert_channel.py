"""AlertChannel protocol — the only interface agents use to send notifications."""

from typing import Protocol
from soccer_forecast_agent.models.match import AlertPayload


class AlertChannel(Protocol):
    """Contract for any notification channel that can deliver an alert payload."""

    def send(self, payload: AlertPayload) -> bool:
        """Deliver the alert. Returns True if delivery succeeded."""
        ...
