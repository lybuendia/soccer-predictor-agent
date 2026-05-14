"""Odds fetcher — retrieves market odds from The Odds API."""

from datetime import datetime, timezone
import uuid
import httpx
from soccer_forecast_agent.domain.team_names import TeamNameNormalizer
from soccer_forecast_agent.models.match import MarketOdds


class OddsFetcher:
    """Fetches decimal market odds from The Odds API and returns typed MarketOdds objects."""

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT_KEY = "soccer_epl"
    PREFERRED_BOOKMAKERS = {"bet365", "williamhill", "unibet", "pinnacle"}

    def __init__(
        self,
        api_key: str,
        client: httpx.Client | None = None,
        team_name_normalizer: TeamNameNormalizer | None = None,
    ) -> None:
        """Initialise with an API key and an optional injected HTTP client."""
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=10)
        self._team_names = team_name_normalizer or TeamNameNormalizer()

    def fetch_odds(self, home_team: str, away_team: str) -> MarketOdds | None:
        """Return the latest market odds for the match identified by home/away team names, or None if not listed."""
        events = self._fetch_all_events()
        event = self._find_event(events, home_team, away_team)
        if event is None:
            return None
        # Use the event's own team name strings for bookmaker outcome lookup — they match the API's outcome names.
        return self._parse_odds(event, event["home_team"], event["away_team"])

    def fetch_all_upcoming_odds(self) -> list[MarketOdds]:
        """Return MarketOdds for all currently listed EPL events."""
        events = self._fetch_all_events()
        results = []
        for event in events:
            odds = self._parse_odds(event, event["home_team"], event["away_team"])
            if odds:
                results.append(odds)
        return results

    def _fetch_all_events(self) -> list[dict]:
        """Fetch all upcoming EPL events with h2h and totals markets from The Odds API."""
        response = self._client.get(
            f"{self.BASE_URL}/sports/{self.SPORT_KEY}/odds/",
            params={
                "apiKey": self._api_key,
                "regions": "uk",
                "markets": "h2h,totals",
                "oddsFormat": "decimal",
            },
        )
        response.raise_for_status()
        return response.json()

    def _find_event(self, events: list[dict], home_team: str, away_team: str) -> dict | None:
        """Find the event matching the given team names using normalised string comparison."""
        home_norm = self._normalise(home_team)
        away_norm = self._normalise(away_team)
        for event in events:
            if self._normalise(event["home_team"]) == home_norm and self._normalise(event["away_team"]) == away_norm:
                return event
        return None

    def _parse_odds(self, event: dict, home_team: str, away_team: str) -> MarketOdds | None:
        """Extract the best available h2h and totals odds from the bookmaker list."""
        h2h = self._best_h2h(event.get("bookmakers", []), home_team, away_team)
        totals = self._best_totals(event.get("bookmakers", []))

        if not h2h or not totals:
            return None

        return MarketOdds(
            odds_id=str(uuid.uuid4()),
            match_id=event["id"],
            timestamp=datetime.now(timezone.utc),
            home_win=h2h["home"],
            draw=h2h["draw"],
            away_win=h2h["away"],
            over_2_5=totals["over"],
            under_2_5=totals["under"],
        )

    def _best_h2h(self, bookmakers: list[dict], home_team: str, away_team: str) -> dict | None:
        """Return h2h odds from the highest-priority available bookmaker."""
        for bm in self._ranked_bookmakers(bookmakers):
            market = next((m for m in bm.get("markets", []) if m["key"] == "h2h"), None)
            if not market:
                continue
            outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
            home = outcomes.get(home_team)
            away = outcomes.get(away_team)
            draw = outcomes.get("Draw")
            if home and away and draw:
                return {"home": home, "draw": draw, "away": away}
        return None

    def _best_totals(self, bookmakers: list[dict]) -> dict | None:
        """Return over/under 2.5 goals odds from the highest-priority available bookmaker."""
        for bm in self._ranked_bookmakers(bookmakers):
            market = next((m for m in bm.get("markets", []) if m["key"] == "totals"), None)
            if not market:
                continue
            for outcome in market["outcomes"]:
                if outcome.get("point") == 2.5:
                    over = next((o["price"] for o in market["outcomes"] if o["name"] == "Over" and o.get("point") == 2.5), None)
                    under = next((o["price"] for o in market["outcomes"] if o["name"] == "Under" and o.get("point") == 2.5), None)
                    if over and under:
                        return {"over": over, "under": under}
        return None

    def _ranked_bookmakers(self, bookmakers: list[dict]) -> list[dict]:
        """Return bookmakers sorted so preferred ones come first."""
        preferred = [b for b in bookmakers if b["key"] in self.PREFERRED_BOOKMAKERS]
        others = [b for b in bookmakers if b["key"] not in self.PREFERRED_BOOKMAKERS]
        return preferred + others

    def _normalise(self, name: str) -> str:
        """Lowercase and strip team name for fuzzy matching across API naming conventions."""
        return self._team_names.canonicalize(name).lower().strip()
