"""
RAG pre-population — fetches recent news for upcoming PL clubs and loads it into ChromaDB.

Run this before a forecast scan to warm the vector store with fresh team context. The
forecast pipeline's ReAct agent will then find relevant prior articles at retrieval time
rather than relying solely on live search results.

Usage:
    source .venv/bin/activate
    python scripts/populate_rag.py [--days-ahead N] [--days-back N] [--all-teams]

Options:
    --days-ahead N   Look-ahead window for upcoming fixtures (default: 7).
                     Teams not playing within this window are skipped unless --all-teams is set.
    --days-back  N   How many days of news to search per team query (default: 3).
    --all-teams      Ingest news for all 20 PL clubs regardless of upcoming fixtures.

Cron example (daily at 06:00):
    0 6 * * * cd /path/to/SoccerForecastAgent && .venv/bin/python scripts/populate_rag.py >> logs/rag.log 2>&1
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import chromadb
from dotenv import load_dotenv

load_dotenv()

from soccer_forecast_agent.config import Config
from soccer_forecast_agent.memory.chroma_repository import ChromaVectorRepository
from soccer_forecast_agent.providers.embeddings import (
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from soccer_forecast_agent.tools.fixtures import FixtureFetcher
from soccer_forecast_agent.tools.ingester import ArticleIngester
from soccer_forecast_agent.tools.search import WebSearchTool
from soccer_forecast_agent.main import build_embedding_provider

ALL_PL_CLUBS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Ipswich",
    "Leicester",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle",
    "Nottingham Forest",
    "Southampton",
    "Sunderland",
    "Tottenham",
    "West Ham",
    "Wolves",
]


def resolve_teams(fixture_fetcher: FixtureFetcher, days_ahead: int, all_teams: bool) -> list[str]:
    """Return the list of team names to ingest for this run."""
    if all_teams:
        return ALL_PL_CLUBS

    try:
        fixtures = fixture_fetcher.fetch_upcoming(competition="PL", days_ahead=days_ahead)
    except Exception as exc:
        print(f"[WARN] Could not fetch upcoming fixtures ({exc}). Falling back to full club list.")
        return ALL_PL_CLUBS

    if not fixtures:
        print("[WARN] No upcoming fixtures found in the next "
              f"{days_ahead} days. Falling back to full club list.")
        return ALL_PL_CLUBS

    seen: set[str] = set()
    teams: list[str] = []
    for match in fixtures:
        for name in (match.home_team, match.away_team):
            if name not in seen:
                seen.add(name)
                teams.append(name)
    return teams


def run(days_ahead: int, days_back: int, all_teams: bool) -> None:
    """Execute the RAG pre-population run."""
    config = Config.from_env()

    chroma_client = chromadb.PersistentClient(path=config.chroma_path)
    embedding_provider = build_embedding_provider(config)
    vector_repo = ChromaVectorRepository(client=chroma_client, embedding_provider=embedding_provider)

    search_tool = WebSearchTool(config.search_api_key)
    fixture_fetcher = FixtureFetcher(config.football_data_api_key)
    ingester = ArticleIngester(search_tool, vector_repo)

    teams = resolve_teams(fixture_fetcher, days_ahead, all_teams)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n[{ts}] Populating RAG — {len(teams)} team(s), {days_back}-day news window\n")

    total_chunks = 0
    for i, team in enumerate(teams, 1):
        try:
            n = ingester.ingest([team], days_back=days_back)
            total_chunks += n
            print(f"  [{i:2}/{len(teams)}] {team:<30} {n:3} chunk(s) stored")
        except Exception as exc:
            print(f"  [{i:2}/{len(teams)}] {team:<30} ERROR: {exc}")

    print(f"\n  Total chunks stored : {total_chunks}")
    print(f"  Teams processed     : {len(teams)}")
    print()


def main() -> None:
    """Parse CLI args and run populate_rag."""
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Pre-populate ChromaDB with recent PL team news.")
    parser.add_argument("--days-ahead", type=int, default=7,
                        help="Upcoming fixture window — only teams playing soon are ingested (default: 7)")
    parser.add_argument("--days-back", type=int, default=3,
                        help="News lookback window per team query (default: 3)")
    parser.add_argument("--all-teams", action="store_true",
                        help="Ingest all 20 PL clubs regardless of upcoming fixtures")
    args = parser.parse_args()
    run(days_ahead=args.days_ahead, days_back=args.days_back, all_teams=args.all_teams)


if __name__ == "__main__":
    main()
