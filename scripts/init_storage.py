"""Initialise local SQLite and Chroma storage for manual validation."""

from pathlib import Path
import sqlite3

import chromadb
from dotenv import load_dotenv

from soccer_forecast_agent.config import Config
from soccer_forecast_agent.main import build_embedding_provider
from soccer_forecast_agent.memory.chroma_repository import ChromaVectorRepository
from soccer_forecast_agent.memory.seed_sources import seed_source_reliability
from soccer_forecast_agent.memory.sqlite_repository import init_db


def main() -> None:
    """Create local SQLite and Chroma storage and print where they live."""
    load_dotenv()
    config = Config.from_env()

    db_path = Path(config.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    init_db(connection)
    seed_source_reliability(connection)
    connection.close()

    chroma_path = Path(config.chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    repo = ChromaVectorRepository(client=client, embedding_provider=build_embedding_provider(config))

    print(f"SQLite initialized at: {db_path.resolve()}")
    print("Source reliability seeded.")
    print(f"Chroma persistent path: {chroma_path.resolve()}")
    print(f"Chroma collection ready: {repo.COLLECTION_NAME}")


if __name__ == "__main__":
    main()
