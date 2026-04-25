"""Fixture fetcher — retrieves upcoming and finished Premier League matches from football-data.org."""

from datetime import date, timedelta
import httpx
from soccer_forecast_agent.domain.team_names import TeamNameNormalizer
from soccer_forecast_agent.models.match import Match
from datetime import datetime


class FixtureFetcher:
    """Fetches upcoming fixtures from football-data.org v4 and returns typed Match objects."""

    BASE_URL = "https://api.football-data.org/v4"
    PL_COMPETITION_ID = "PL"

    def __init__(
        self,
        api_key: str,
        client: httpx.Client | None = None,
        team_name_normalizer: TeamNameNormalizer | None = None,
    ) -> None:
        """Initialise with an API key and an optional injected HTTP client."""
        self._api_key = self._validate_api_key(api_key)
        self._client = client or httpx.Client(timeout=10)
        self._team_names = team_name_normalizer or TeamNameNormalizer()

    def fetch_upcoming(self, competition: str = PL_COMPETITION_ID, days_ahead: int = 7) -> list[Match]:
        """Return upcoming fixtures for the competition within the next days_ahead days."""
        today = date.today()
        date_to = today + timedelta(days=days_ahead)
        return self._fetch_matches(
            competition=competition,
            status="SCHEDULED",
            date_from=today.isoformat(),
            date_to=date_to.isoformat(),
        )

    def fetch_finished(self, competition: str = PL_COMPETITION_ID, days_back: int = 365) -> list[Match]:
        """Return finished fixtures for the competition within the previous days_back days."""
        today = date.today()
        date_from = today - timedelta(days=days_back)
        return self._fetch_matches(
            competition=competition,
            status="FINISHED",
            date_from=date_from.isoformat(),
            date_to=today.isoformat(),
        )

    def _fetch_matches(self, competition: str, status: str, date_from: str, date_to: str) -> list[Match]:
        """Fetch matches for a competition and date window from football-data.org."""
        response = self._client.get(
            f"{self.BASE_URL}/competitions/{competition}/matches",
            headers={"X-Auth-Token": self._api_key},
            params={
                "status": status,
                "dateFrom": date_from,
                "dateTo": date_to,
            },
        )
        response.raise_for_status()
        return [self._parse_match(m, competition) for m in response.json().get("matches", [])]

    def _parse_match(self, raw: dict, competition: str) -> Match:
        """Convert a raw API match dict to a typed Match."""
        full_time = raw.get("score", {}).get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")
        is_finished = raw.get("status") == "FINISHED" and home_goals is not None and away_goals is not None
        return Match(
            match_id=str(raw["id"]),
            competition=competition,
            home_team=self._team_names.canonicalize(raw["homeTeam"]["name"]),
            away_team=self._team_names.canonicalize(raw["awayTeam"]["name"]),
            kickoff_time=datetime.fromisoformat(raw["utcDate"].replace("Z", "+00:00")),
            status="resolved" if is_finished else "upcoming",
            final_score=f"{home_goals}-{away_goals}" if is_finished else None,
        )

    def _validate_api_key(self, api_key: str) -> str:
        """Return a clean API key or raise a clear error when it is missing."""
        if not isinstance(api_key, str):
            raise ValueError("FOOTBALL_DATA_API_KEY is missing or invalid in the environment.")
        cleaned = api_key.strip()
        if not cleaned:
            raise ValueError("FOOTBALL_DATA_API_KEY is empty. Add your football-data.org API key to .env.")
        return cleaned
