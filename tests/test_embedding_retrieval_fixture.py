import json
from pathlib import Path

import chromadb

from soccer_forecast_agent.memory.chroma_repository import ChromaVectorRepository
from soccer_forecast_agent.models.evidence import ArticleChunk


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_news_articles.json"


class FakeKeywordEmbeddingProvider:
    """Deterministic embedding stub for retrieval tests without model downloads."""

    KEYWORDS = [
        "arsenal",
        "chelsea",
        "manchester city",
        "manchester united",
        "tottenham",
        "injur",
        "title",
        "relegation",
        "champions league",
        "brighton",
    ]

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            embeddings.append([1.0 if keyword in lowered else 0.0 for keyword in self.KEYWORDS])
        return embeddings


def _load_fixture_chunks() -> list[ArticleChunk]:
    articles = json.loads(FIXTURE_PATH.read_text())
    return [
        ArticleChunk(
            chunk_id=article["article_id"],
            content=f"{article['title']}. {article['content']}",
            source=article["source"],
            url=article["url"],
            published_at=article["published_at"],
            teams=article["teams"],
        )
        for article in articles
    ]


def test_sample_fixture_contains_six_articles():
    chunks = _load_fixture_chunks()

    assert len(chunks) == 6


def test_retrieval_fixture_surfaces_expected_team_and_topic_matches():
    repo = ChromaVectorRepository(
        client=chromadb.EphemeralClient(),
        embedding_provider=FakeKeywordEmbeddingProvider(),
    )
    repo.upsert(_load_fixture_chunks())

    arsenal_results = repo.search("Arsenal injuries", teams=["Arsenal"], top_k=3)
    united_results = repo.search("Manchester United Champions League race", teams=["Manchester United"], top_k=3)
    spurs_results = repo.search("Tottenham relegation danger", teams=["Tottenham"], top_k=3)

    assert arsenal_results
    assert "Arsenal" in arsenal_results[0].teams
    assert "injur" in arsenal_results[0].content.lower()

    assert united_results
    assert "Manchester United" in united_results[0].teams
    assert "champions league" in united_results[0].content.lower()

    assert spurs_results
    assert "Tottenham" in spurs_results[0].teams
    assert "relegation" in spurs_results[0].content.lower()
