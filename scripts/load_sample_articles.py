"""Load six web-sourced sample article summaries into Chroma for retrieval checks."""

import json
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from soccer_forecast_agent.config import Config
from soccer_forecast_agent.main import build_embedding_provider
from soccer_forecast_agent.memory.chroma_repository import ChromaVectorRepository
from soccer_forecast_agent.models.evidence import ArticleChunk


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_news_articles.json"


def main() -> None:
    """Load the local sample article fixture into persistent Chroma storage."""
    load_dotenv()
    config = Config.from_env()

    articles = json.loads(FIXTURE_PATH.read_text())
    client = chromadb.PersistentClient(path=config.chroma_path)
    repo = ChromaVectorRepository(client=client, embedding_provider=build_embedding_provider(config))

    chunks = [
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

    repo.upsert(chunks)
    print(f"Loaded {len(chunks)} sample articles into Chroma.")


if __name__ == "__main__":
    main()
