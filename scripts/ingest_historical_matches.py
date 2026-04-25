"""Fetch recent finished Premier League matches and store them in SQLite."""

import sqlite3

from dotenv import load_dotenv

from soccer_forecast_agent.config import Config
from soccer_forecast_agent.memory.seed_sources import seed_source_reliability
from soccer_forecast_agent.memory.sqlite_repository import SQLiteRepository, init_db
from soccer_forecast_agent.tools.fixtures import FixtureFetcher


def main(days_back: int = 365) -> None:
    """Fetch finished matches for the last days_back days and persist them locally."""
    load_dotenv()
    config = Config.from_env()

    connection = sqlite3.connect(config.db_path)
    init_db(connection)
    seed_source_reliability(connection)
    repository = SQLiteRepository(connection)

    fetcher = FixtureFetcher(api_key=config.football_data_api_key)
    matches = fetcher.fetch_finished(competition="PL", days_back=days_back)
    for match in matches:
        repository.save_match(match)

    connection.close()
    print(f"Ingested {len(matches)} finished matches into SQLite.")


if __name__ == "__main__":
    main()
