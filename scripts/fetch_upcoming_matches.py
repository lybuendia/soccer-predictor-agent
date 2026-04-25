"""Fetch upcoming Premier League fixtures, print them, and optionally store them in SQLite."""

import argparse
import sqlite3

from dotenv import load_dotenv

from soccer_forecast_agent.config import Config
from soccer_forecast_agent.memory.seed_sources import seed_source_reliability
from soccer_forecast_agent.memory.sqlite_repository import SQLiteRepository, init_db
from soccer_forecast_agent.tools.fixtures import FixtureFetcher


def main() -> None:
    """Fetch upcoming fixtures for the next N days and optionally save them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-ahead", type=int, default=7)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    config = Config.from_env()
    fetcher = FixtureFetcher(api_key=config.football_data_api_key)
    matches = fetcher.fetch_upcoming(competition="PL", days_ahead=args.days_ahead)

    print(f"Fetched {len(matches)} upcoming matches for the next {args.days_ahead} days.")
    for match in matches:
        print(
            f"{match.kickoff_time.isoformat()} | {match.home_team} vs {match.away_team} | "
            f"status={match.status} | id={match.match_id}"
        )

    if not args.save:
        return

    connection = sqlite3.connect(config.db_path)
    init_db(connection)
    seed_source_reliability(connection)
    repository = SQLiteRepository(connection)
    for match in matches:
        repository.save_match(match)
    connection.close()
    print(f"Saved {len(matches)} upcoming matches into SQLite at {config.db_path}.")


if __name__ == "__main__":
    main()
