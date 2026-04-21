"""Abstract repository protocols for all persistence layers."""

from typing import Protocol
from soccer_forecast_agent.models.match import Match, MarketOdds, Forecast
from soccer_forecast_agent.models.evidence import EvidenceItem, ArticleChunk


class MatchRepository(Protocol):
    """Persistence contract for match fixtures."""

    def save_match(self, match: Match) -> None:
        """Persist a match, upserting on match_id."""
        ...

    def get_upcoming(self, competition: str) -> list[Match]:
        """Return all matches with status 'upcoming' for the given competition."""
        ...

    def update_status(self, match_id: str, status: str, final_score: str | None = None) -> None:
        """Update match status and optionally record the final score."""
        ...


class ForecastRepository(Protocol):
    """Persistence contract for forecast records."""

    def save_forecast(self, forecast: Forecast) -> None:
        """Persist a forecast record."""
        ...

    def get_by_match(self, match_id: str) -> list[Forecast]:
        """Return all forecasts for a given match, ordered by run_timestamp descending."""
        ...

    def get_latest(self, match_id: str) -> Forecast | None:
        """Return the most recent forecast for a match, or None if none exists."""
        ...


class EvidenceRepository(Protocol):
    """Persistence contract for evidence items."""

    def save_evidence(self, item: EvidenceItem) -> None:
        """Persist a single evidence item."""
        ...

    def get_by_forecast(self, forecast_id: str) -> list[EvidenceItem]:
        """Return all evidence items linked to a forecast."""
        ...


class SourceReliabilityRepository(Protocol):
    """Persistence contract for source reliability scores."""

    def get_score(self, source_domain: str) -> float:
        """Return the reliability score (0.0–1.0) for a source domain; defaults to 0.5 if unknown."""
        ...


class VectorRepository(Protocol):
    """Persistence contract for the unstructured news vector store."""

    def upsert(self, chunks: list[ArticleChunk]) -> None:
        """Embed and store article chunks, replacing any with the same chunk_id."""
        ...

    def search(self, query: str, teams: list[str], top_k: int = 5) -> list[ArticleChunk]:
        """Return the top_k most semantically relevant chunks filtered by team names."""
        ...
