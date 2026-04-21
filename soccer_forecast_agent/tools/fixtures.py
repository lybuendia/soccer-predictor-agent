"""Fixture fetcher — retrieves upcoming Premier League matches from football-data.org."""

from datetime import date, timedelta
import httpx
from soccer_forecast_agent.models.match import Match
from datetime import datetime, timezone


class FixtureFetcher:
    """Fetches upcoming fixtures from football-data.org v4 and returns typed Match objects."""

    BASE_URL = "https://api.football-data.org/v4"
    PL_COMPETITION_ID = "PL"

    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        """Initialise with an API key and an optional injected HTTP client."""
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=10)

    def fetch_upcoming(self, competition: str = PL_COMPETITION_ID, days_ahead: int = 7) -> list[Match]:
        """Return upcoming fixtures for the competition within the next days_ahead days."""
        today = date.today()
        date_to = today + timedelta(days=days_ahead)

        response = self._client.get(
            f"{self.BASE_URL}/competitions/{competition}/matches",
            headers={"X-Auth-Token": self._api_key},
            params={
                "status": "SCHEDULED",
                "dateFrom": today.isoformat(),
                "dateTo": date_to.isoformat(),
            },
        )
        response.raise_for_status()

        return [self._parse_match(m, competition) for m in response.json().get("matches", [])]

    def _parse_match(self, raw: dict, competition: str) -> Match:
        """Convert a raw API match dict to a typed Match."""
        return Match(
            match_id=str(raw["id"]),
            competition=competition,
            home_team=raw["homeTeam"]["name"],
            away_team=raw["awayTeam"]["name"],
            kickoff_time=datetime.fromisoformat(raw["utcDate"].replace("Z", "+00:00")),
            status="upcoming",
            final_score=None,
        )
