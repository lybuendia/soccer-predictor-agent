"""Alert channel implementations — SMTP email and console fallback."""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from soccer_forecast_agent.models.match import AlertPayload


class EmailAlertChannel:
    """Implements AlertChannel by sending emails via SMTP with STARTTLS."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        recipient: str,
    ) -> None:
        """Initialise with SMTP credentials and recipient address."""
        self._host = smtp_host
        self._port = smtp_port
        self._user = smtp_user
        self._password = smtp_password
        self._recipient = recipient

    def send(self, payload: AlertPayload) -> bool:
        """Compose and send the alert email. Returns True if delivery succeeded, False on SMTP error."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = self._subject(payload)
        msg["From"] = self._user
        msg["To"] = self._recipient

        msg.attach(MIMEText(self._render_body(payload), "plain"))

        try:
            with smtplib.SMTP(self._host, self._port) as server:
                server.ehlo()
                server.starttls()
                server.login(self._user, self._password)
                server.sendmail(self._user, self._recipient, msg.as_string())
            return True
        except smtplib.SMTPException:
            return False

    def _subject(self, payload: AlertPayload) -> str:
        """Build the email subject line from match and edge details."""
        m = payload.match
        edge = payload.forecast.edge_value or 0.0
        market = payload.forecast.edge_market or "unknown"
        return f"[SoccerForecast] Edge Alert: {m.home_team} vs {m.away_team} — {market} ({edge:+.1%})"

    def _render_body(self, payload: AlertPayload) -> str:
        """Render the alert payload as a structured plain-text email body."""
        m = payload.match
        f = payload.forecast
        b = f.baseline

        lines = [
            f"MATCH:    {m.home_team} vs {m.away_team}",
            f"KICKOFF:  {m.kickoff_time.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "BASELINE PROBABILITIES",
            f"  Home Win : {b.home_win:.1%}",
            f"  Draw     : {b.draw:.1%}",
            f"  Away Win : {b.away_win:.1%}",
            f"  Over 2.5 : {b.over_2_5:.1%}",
            f"  Under 2.5: {b.under_2_5:.1%}",
            "",
        ]

        if f.adjusted_home_win is not None:
            lines += [
                "ADJUSTED PROBABILITIES",
                f"  Home Win : {f.adjusted_home_win:.1%}",
                f"  Draw     : {f.adjusted_draw:.1%}",
                f"  Away Win : {f.adjusted_away_win:.1%}",
                f"  Over 2.5 : {f.adjusted_over_2_5:.1%}",
                f"  Under 2.5: {f.adjusted_under_2_5:.1%}",
                "",
            ]

        lines += [
            f"EDGE DETECTED: {f.edge_market or 'unknown'} ({(f.edge_value or 0.0):+.1%})",
            f"CONFIDENCE   : {f.confidence_score:.1%}",
            "",
            "RATIONALE",
            payload.rationale,
            "",
            "-" * 60,
            payload.disclaimer,
        ]

        return "\n".join(lines)


class ConsoleAlertChannel:
    """Prints alert payloads to stdout — used when SMTP is not configured."""

    def send(self, payload: AlertPayload) -> bool:
        """Print the alert to stdout and return True."""
        m = payload.match
        f = payload.forecast
        print(f"\n{'=' * 60}")
        print(f"ALERT: {m.home_team} vs {m.away_team}")
        print(f"Edge: {f.edge_market} ({(f.edge_value or 0.0):+.1%})  |  Confidence: {f.confidence_score:.1%}")
        print(f"Adjusted: home={f.adjusted_home_win:.1%}  draw={f.adjusted_draw:.1%}  away={f.adjusted_away_win:.1%}")
        print(f"Rationale: {payload.rationale}")
        print(f"{'=' * 60}\n")
        return True
