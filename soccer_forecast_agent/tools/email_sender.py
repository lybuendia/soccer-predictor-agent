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

        msg.attach(MIMEText(self._render_plain(payload), "plain"))
        msg.attach(MIMEText(self._render_html(payload), "html"))

        try:
            with smtplib.SMTP(self._host, self._port) as server:
                server.ehlo()
                server.starttls()
                server.login(self._user, self._password)
                server.sendmail(self._user, self._recipient, msg.as_string())
            return True
        except smtplib.SMTPException:
            return False

    # ------------------------------------------------------------------
    # Subject
    # ------------------------------------------------------------------

    def _subject(self, payload: AlertPayload) -> str:
        """Build the email subject line from match and edge details."""
        m = payload.match
        edge = payload.forecast.edge_value or 0.0
        market = _market_label(payload, payload.forecast.edge_market)
        return (
            f"⚽ SoccerForecast — {m.home_team} vs {m.away_team}"
            f" | {market} edge {edge:+.1%}"
        )

    # ------------------------------------------------------------------
    # HTML body
    # ------------------------------------------------------------------

    def _render_html(self, payload: AlertPayload) -> str:
        """Render the alert as a styled HTML email."""
        m = payload.match
        f = payload.forecast
        b = f.baseline
        odds = _odds_for_payload(payload)

        edge = f.edge_value or 0.0
        market_label = _market_label(payload, f.edge_market)
        kickoff = m.kickoff_time.strftime("%A, %d %b %Y · %H:%M UTC")
        edge_color = "#16a34a" if edge >= 0.07 else "#d97706"
        conf_pct = int(f.confidence_score * 100)
        conf_color = "#16a34a" if f.confidence_score >= 0.75 else (
            "#d97706" if f.confidence_score >= 0.55 else "#dc2626"
        )

        adjusted = f.adjusted_home_win is not None

        def prob_row(label: str, base: float, adj: float | None) -> str:
            adj_cell = (
                f'<td style="padding:6px 12px;text-align:right;font-weight:600;'
                f'color:{edge_color if adj and adj > base else "#374151"};">'
                f'{adj:.1%}</td>'
            ) if adj is not None else ""
            return (
                f'<tr style="border-bottom:1px solid #f3f4f6;">'
                f'<td style="padding:6px 12px;color:#6b7280;">{label}</td>'
                f'<td style="padding:6px 12px;text-align:right;">{base:.1%}</td>'
                f'{adj_cell}'
                f'</tr>'
            )

        adj_header = (
            '<th style="padding:6px 12px;text-align:right;color:#6b7280;'
            'font-weight:500;font-size:12px;">Adjusted</th>'
        ) if adjusted else ""

        rows = "".join([
            prob_row(
                f"{m.home_team} win",
                b.home_win,
                f.adjusted_home_win if adjusted else None,
            ),
            prob_row("Draw", b.draw, f.adjusted_draw if adjusted else None),
            prob_row(
                f"{m.away_team} win",
                b.away_win,
                f.adjusted_away_win if adjusted else None,
            ),
            prob_row(
                "Over 2.5 goals",
                b.over_2_5,
                f.adjusted_over_2_5 if adjusted else None,
            ),
            prob_row(
                "Under 2.5 goals",
                b.under_2_5,
                f.adjusted_under_2_5 if adjusted else None,
            ),
        ])

        source_text = _market_source_summary(odds)

        return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:20px 0;background:#f1f5f9;font-family:Arial,Helvetica,sans-serif;">
  <div style="max-width:600px;margin:0 auto;">

    <!-- Header -->
    <div style="background:#0f172a;border-radius:12px 12px 0 0;padding:28px 28px 22px;">
      <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#64748b;margin-bottom:6px;">
        &#9917; Soccer Forecast Alert
      </div>
      <div style="font-size:24px;font-weight:700;color:#f8fafc;line-height:1.2;">
        {m.home_team}
        <span style="color:#475569;font-weight:400;font-size:18px;"> vs </span>
        {m.away_team}
      </div>
      <div style="margin-top:8px;font-size:13px;color:#94a3b8;">
        &#128197; {kickoff}
      </div>
    </div>

    <!-- Edge banner -->
    <div style="background:{edge_color};padding:16px 28px;">
      <div style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;
                  color:rgba(255,255,255,0.75);margin-bottom:4px;">Value Signal</div>
      <div style="display:flex;align-items:center;gap:12px;">
        <span style="font-size:20px;font-weight:700;color:#fff;">{market_label}</span>
        <span style="background:rgba(255,255,255,0.2);color:#fff;font-size:15px;
                     font-weight:700;padding:3px 10px;border-radius:20px;">{edge:+.1%}</span>
      </div>
      <div style="margin-top:6px;font-size:13px;color:rgba(255,255,255,0.88);">
        {_edge_explanation(payload)}
      </div>
    </div>

    <!-- Probabilities card -->
    <div style="background:#fff;padding:0;">
      <div style="padding:16px 28px 8px;font-size:12px;font-weight:600;
                  text-transform:uppercase;letter-spacing:1px;color:#6b7280;">
        Probabilities
      </div>
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
            <th style="padding:8px 12px;text-align:left;color:#6b7280;
                       font-weight:500;font-size:12px;">Market</th>
            <th style="padding:8px 12px;text-align:right;color:#6b7280;
                       font-weight:500;font-size:12px;">Baseline</th>
            {adj_header}
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>

    <!-- Confidence + source row -->
    <div style="background:#fff;border-top:1px solid #f1f5f9;
                padding:16px 28px;display:flex;gap:24px;flex-wrap:wrap;">
      <div style="flex:1;min-width:200px;">
        <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;
                    color:#6b7280;margin-bottom:6px;">Confidence</div>
        <div style="background:#e2e8f0;border-radius:4px;height:8px;overflow:hidden;">
          <div style="background:{conf_color};width:{conf_pct}%;height:100%;
                      border-radius:4px;transition:width 0.3s;"></div>
        </div>
        <div style="margin-top:4px;font-size:13px;font-weight:600;color:{conf_color};">
          {f.confidence_score:.0%}
        </div>
      </div>
      <div style="flex:2;min-width:200px;">
        <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;
                    color:#6b7280;margin-bottom:6px;">Odds Source</div>
        <div style="font-size:13px;color:#374151;">{source_text}</div>
      </div>
    </div>

    <!-- Rationale -->
    <div style="background:#fff;border-top:1px solid #f1f5f9;padding:16px 28px 20px;">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;
                  color:#6b7280;margin-bottom:8px;">Rationale</div>
      <div style="font-size:14px;color:#374151;line-height:1.6;">
        {payload.rationale}
      </div>
    </div>

    <!-- Disclaimer footer -->
    <div style="background:#f8fafc;border-radius:0 0 12px 12px;padding:14px 28px;
                border-top:1px solid #e2e8f0;">
      <div style="font-size:11px;color:#9ca3af;line-height:1.5;">
        {payload.disclaimer}
      </div>
    </div>

  </div>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Plain-text fallback
    # ------------------------------------------------------------------

    def _render_plain(self, payload: AlertPayload) -> str:
        """Render the alert as plain text (fallback for clients that don't render HTML)."""
        m = payload.match
        f = payload.forecast
        b = f.baseline
        odds = _odds_for_payload(payload)

        lines = [
            f"{'=' * 60}",
            f"  SOCCER FORECAST ALERT",
            f"{'=' * 60}",
            f"  {m.home_team} vs {m.away_team}",
            f"  {m.kickoff_time.strftime('%A, %d %b %Y · %H:%M UTC')}",
            "",
            f"  VALUE SIGNAL : {_market_label(payload, f.edge_market)}  ({f.edge_value or 0:+.1%})",
            f"  {_edge_explanation(payload)}",
            "",
            "  PROBABILITIES",
            f"  {'Market':<22} {'Baseline':>9}" + (
                f"  {'Adjusted':>9}" if f.adjusted_home_win is not None else ""
            ),
            f"  {'-'*22} {'-'*9}" + (f"  {'-'*9}" if f.adjusted_home_win is not None else ""),
        ]

        def prow(label, base, adj):
            row = f"  {label:<22} {base:>8.1%}"
            if adj is not None:
                row += f"  {adj:>8.1%}"
            return row

        adj = f.adjusted_home_win is not None
        lines += [
            prow(f"{m.home_team} win", b.home_win, f.adjusted_home_win if adj else None),
            prow("Draw", b.draw, f.adjusted_draw if adj else None),
            prow(f"{m.away_team} win", b.away_win, f.adjusted_away_win if adj else None),
            prow("Over 2.5 goals", b.over_2_5, f.adjusted_over_2_5 if adj else None),
            prow("Under 2.5 goals", b.under_2_5, f.adjusted_under_2_5 if adj else None),
            "",
            f"  CONFIDENCE   : {f.confidence_score:.0%}",
            f"  ODDS SOURCE  : {_market_source_summary(odds)}",
            "",
            "  RATIONALE",
            f"  {payload.rationale}",
            "",
            f"  {'-' * 56}",
            f"  {payload.disclaimer}",
        ]
        return "\n".join(lines)


class ConsoleAlertChannel:
    """Prints alert payloads to stdout — used when SMTP is not configured."""

    def send(self, payload: AlertPayload) -> bool:
        """Print the alert to stdout and return True."""
        m = payload.match
        f = payload.forecast
        odds = _odds_for_payload(payload)
        b = f.baseline

        w = 64
        print(f"\n{'=' * w}")
        print(f"  ⚽  SOCCER FORECAST ALERT")
        print(f"{'=' * w}")
        print(f"  {m.home_team} vs {m.away_team}")
        print(f"  {m.kickoff_time.strftime('%A, %d %b %Y · %H:%M UTC')}")
        print(f"{'─' * w}")
        print(f"  VALUE SIGNAL  : {_market_label(payload, f.edge_market)}  ({f.edge_value or 0:+.1%})")
        print(f"  {_edge_explanation(payload)}")
        print(f"{'─' * w}")
        adj = f.adjusted_home_win is not None
        header = f"  {'Market':<24} {'Baseline':>8}" + (f"  {'Adjusted':>8}" if adj else "")
        print(header)
        print(f"  {'-'*24} {'-'*8}" + (f"  {'-'*8}" if adj else ""))
        for label, base, adjv in [
            (f"{m.home_team} win", b.home_win, f.adjusted_home_win),
            ("Draw", b.draw, f.adjusted_draw),
            (f"{m.away_team} win", b.away_win, f.adjusted_away_win),
            ("Over 2.5 goals", b.over_2_5, f.adjusted_over_2_5),
            ("Under 2.5 goals", b.under_2_5, f.adjusted_under_2_5),
        ]:
            row = f"  {label:<24} {base:>7.1%}"
            if adj and adjv is not None:
                row += f"  {adjv:>7.1%}"
            print(row)
        print(f"{'─' * w}")
        print(f"  CONFIDENCE    : {f.confidence_score:.0%}")
        print(f"  ODDS SOURCE   : {_market_source_summary(odds)}")
        print(f"{'─' * w}")
        print(f"  RATIONALE")
        print(f"  {payload.rationale}")
        print(f"{'=' * w}\n")
        return True


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _market_label(payload: AlertPayload, market: str | None) -> str:
    """Return a human-friendly market label."""
    if market is None:
        return "Unknown market"
    labels = {
        "home_win": f"{payload.match.home_team} win",
        "draw": "Draw",
        "away_win": f"{payload.match.away_team} win",
        "over_2_5": "Over 2.5 goals",
        "under_2_5": "Under 2.5 goals",
    }
    return labels.get(market, market)


def _market_probability(payload: AlertPayload, market: str | None) -> float | None:
    """Return the adjusted probability for the selected edge market."""
    forecast = payload.forecast
    probabilities = {
        "home_win": forecast.adjusted_home_win,
        "draw": forecast.adjusted_draw,
        "away_win": forecast.adjusted_away_win,
        "over_2_5": forecast.adjusted_over_2_5,
        "under_2_5": forecast.adjusted_under_2_5,
    }
    return probabilities.get(market) if market is not None else None


def _edge_explanation(payload: AlertPayload) -> str:
    """One-sentence plain-language description of the detected edge."""
    forecast = payload.forecast
    market = forecast.edge_market
    edge = forecast.edge_value
    adj_prob = _market_probability(payload, market)
    if market is None or edge is None or adj_prob is None:
        return "Edge detail unavailable."
    implied = adj_prob - edge
    return (
        f"We estimate {_market_label(payload, market)} at {adj_prob:.1%}, "
        f"vs market-implied {implied:.1%} — a {edge:+.1%} gap."
    )


def _odds_for_payload(payload: AlertPayload):
    """Return the MarketOdds attached to the forecast if present."""
    return payload.market_odds


def _market_source_summary(odds) -> str:
    """Return a human-friendly summary of the odds sources behind the alert."""
    if odds is None:
        return "Unavailable."
    winner_source = getattr(odds, "winner_market_source", "Unknown source")
    goals_source = getattr(odds, "goals_market_source", "Unknown source")
    seen = list(getattr(odds, "market_sources_seen", []))
    seen_suffix = f" (also seen: {', '.join(seen[:3])})" if seen else ""
    if winner_source == goals_source:
        return f"{winner_source} via The Odds API{seen_suffix}"
    return f"{winner_source} (winner), {goals_source} (goals) via The Odds API{seen_suffix}"
