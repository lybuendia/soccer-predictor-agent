from dataclasses import dataclass

from soccer_forecast_agent.tools.ingester import ArticleIngester
from soccer_forecast_agent.tools.search import SearchResult


class FakeSearchTool:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        self.calls.append((query, max_results))
        return self._results


@dataclass
class FakeVectorRepository:
    stored_chunks: list = None

    def __post_init__(self) -> None:
        if self.stored_chunks is None:
            self.stored_chunks = []

    def upsert(self, chunks: list) -> None:
        self.stored_chunks.extend(chunks)


def test_article_ingester_skips_blank_and_duplicate_urls():
    long_snippet = "Arsenal injury update " * 40
    search_tool = FakeSearchTool(
        [
            SearchResult(url="https://example.com/a", title="A", snippet=long_snippet),
            SearchResult(url="https://example.com/a", title="A duplicate", snippet=long_snippet),
            SearchResult(url="", title="Blank", snippet="Should be ignored"),
            SearchResult(url="https://example.com/b", title="B", snippet="Short preview"),
        ]
    )
    vector_repo = FakeVectorRepository()
    ingester = ArticleIngester(search_tool, vector_repo)

    count = ingester.ingest(["Arsenal", "Chelsea"], days_back=2)

    assert count == len(vector_repo.stored_chunks)
    assert len(vector_repo.stored_chunks) >= 2
    assert all(chunk.url for chunk in vector_repo.stored_chunks)
    assert {chunk.url for chunk in vector_repo.stored_chunks} == {
        "https://example.com/a",
        "https://example.com/b",
    }
    assert search_tool.calls == [("Arsenal Chelsea soccer news injuries lineup last 2 days", 10)]


def test_article_ingester_chunking_uses_overlap():
    ingester = ArticleIngester(FakeSearchTool([]), FakeVectorRepository())
    text = "A" * 500

    chunks = ingester._chunk(text, chunk_size=200)

    assert len(chunks) == 4
    assert chunks[0] == "A" * 200
    assert chunks[1] == "A" * 200
    assert ingester._chunk("") == []
