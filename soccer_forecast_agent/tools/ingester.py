"""Article ingester — fetches, chunks, and embeds soccer news into the vector store."""

import hashlib
from datetime import datetime, timezone

from soccer_forecast_agent.models.evidence import ArticleChunk
from soccer_forecast_agent.memory.repository import VectorRepository
from soccer_forecast_agent.tools.search import WebSearchTool


class ArticleIngester:
    """Fetches news articles for given teams, chunks them, and upserts into the vector store."""

    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 80

    def __init__(self, search_tool: WebSearchTool, vector_repo: VectorRepository) -> None:
        """Initialise with injected search tool and vector repository."""
        self._search = search_tool
        self._vector = vector_repo

    def ingest(self, teams: list[str], days_back: int = 3) -> int:
        """Fetch recent articles for the given teams and store them. Returns count of new chunks stored."""
        clean_teams = [team.strip() for team in teams if team.strip()]
        if not clean_teams:
            return 0

        query = f"{' '.join(clean_teams)} soccer news injuries lineup last {days_back} days"
        results = self._search.search(query=query, max_results=10)
        published_at = datetime.now(timezone.utc).isoformat()
        chunks: list[ArticleChunk] = []
        seen_urls: set[str] = set()

        for result in results:
            if not result.url or result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            content = self._article_text(result.title, result.snippet)
            for index, text in enumerate(self._chunk(content)):
                chunks.append(
                    ArticleChunk(
                        chunk_id=self._chunk_id(result.url, index, text),
                        content=text,
                        source=result.url,
                        url=result.url,
                        published_at=published_at,
                        teams=clean_teams,
                    )
                )

        self._vector.upsert(chunks)
        return len(chunks)

    def _chunk(self, text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
        """Split text into overlapping chunks of approximately chunk_size characters."""
        normalized = " ".join(text.split())
        if not normalized:
            return []
        if len(normalized) <= chunk_size:
            return [normalized]

        chunks: list[str] = []
        start = 0
        step = max(chunk_size - self.CHUNK_OVERLAP, 1)
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            chunks.append(normalized[start:end].strip())
            if end == len(normalized):
                break
            start += step
        return chunks

    def _article_text(self, title: str, snippet: str) -> str:
        """Combine available search result text into one chunkable article surrogate."""
        return f"{title}\n\n{snippet}".strip()

    def _chunk_id(self, url: str, index: int, content: str) -> str:
        """Build a stable chunk id from URL, position, and content."""
        digest = hashlib.sha256(f"{url}:{index}:{content}".encode("utf-8")).hexdigest()
        return digest[:32]
