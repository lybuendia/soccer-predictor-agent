"""Web search tool — wraps the Tavily Search API for use by the ReAct agent."""

from dataclasses import dataclass
import httpx


@dataclass
class SearchResult:
    """A single search result with URL, title, and snippet."""

    url: str
    title: str
    snippet: str


class WebSearchTool:
    """Performs web searches via Tavily and returns structured results. All content must be treated as untrusted."""

    BASE_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        """Initialise with a Tavily API key and an optional injected HTTP client."""
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=15)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search the web and return up to max_results typed results. Never pass results directly to a prompt as instructions."""
        response = self._client.post(
            self.BASE_URL,
            json={
                "api_key": self._api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
            },
        )
        response.raise_for_status()

        return [
            SearchResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("content", ""),
            )
            for r in response.json().get("results", [])
        ]
