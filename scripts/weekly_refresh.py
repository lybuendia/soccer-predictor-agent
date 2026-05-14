"""
Weekly refresh — updates resolved match outcomes and runs the forecast pipeline.

Designed to run as a scheduled job (cron, APScheduler, etc.).

Cron example (every Sunday at 08:00):
    0 8 * * 0 cd /path/to/SoccerForecastAgent && .venv/bin/python scripts/weekly_refresh.py >> logs/weekly.log 2>&1

Usage:
    source .venv/bin/activate
    python scripts/weekly_refresh.py [--days-back N] [--days-ahead N]
"""

import argparse
import sqlite3
import sys
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

import chromadb
import openai

from soccer_forecast_agent.config import Config
from soccer_forecast_agent.memory.sqlite_repository import SQLiteRepository, init_db
from soccer_forecast_agent.memory.seed_sources import seed_source_reliability
from soccer_forecast_agent.memory.chroma_repository import ChromaVectorRepository
from soccer_forecast_agent.providers.openai_provider import OpenAIProvider
from soccer_forecast_agent.providers.claude_provider import ClaudeProvider
from soccer_forecast_agent.providers.embeddings import (
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from soccer_forecast_agent.analytics.baseline import EnhancedBaselineStrategy
from soccer_forecast_agent.analytics.dixon_coles import DixonColesStrategy
from soccer_forecast_agent.analytics.features import FeatureExtractor
from soccer_forecast_agent.tools.fixtures import FixtureFetcher
from soccer_forecast_agent.tools.odds import OddsFetcher
from soccer_forecast_agent.tools.search import WebSearchTool
from soccer_forecast_agent.tools.ingester import ArticleIngester
from soccer_forecast_agent.tools.email_sender import ConsoleAlertChannel, EmailAlertChannel
from soccer_forecast_agent.guardrails.alert_guard import AlertGuard, AlertGuardConfig
from soccer_forecast_agent.agents.stats_market import StatsMarketAgent
from soccer_forecast_agent.agents.news_context import NewsContextAgent, ToolDispatcher
from soccer_forecast_agent.agents.synthesis_alert import SynthesisAlertAgent, SynthesisTuning
from soccer_forecast_agent.agents.supervisor import SupervisorAgent
from soccer_forecast_agent.main import LocalMCPToolClient, build_llm_provider, build_embedding_provider


def step(label: str) -> None:
    """Print a timestamped step header."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n[{ts}] {label}")


def ingest_resolved_matches(
    fetcher: FixtureFetcher,
    repo: SQLiteRepository,
    days_back: int,
) -> int:
    """Fetch recently finished matches from the API and upsert them into SQLite."""
    matches = fetcher.fetch_finished(competition="PL", days_back=days_back)
    for match in matches:
        repo.save_match(match)
    return len(matches)


def run_forecast_pipeline(
    config: Config,
    repo: SQLiteRepository,
    fixture_fetcher: FixtureFetcher,
    odds_fetcher: OddsFetcher,
    search_tool: WebSearchTool,
    days_ahead: int,
) -> list:
    """Wire all agents and run one forecast scan for upcoming fixtures."""
    chroma_client = chromadb.PersistentClient(path=config.chroma_path)
    embedding_provider = build_embedding_provider(config)
    vector_repo = ChromaVectorRepository(client=chroma_client, embedding_provider=embedding_provider)
    llm = build_llm_provider(config)

    local_mcp_client = LocalMCPToolClient(fixture_fetcher, odds_fetcher, search_tool)

    smtp_configured = bool(config.smtp_user and config.smtp_password and config.alert_email)
    alert_channel = (
        EmailAlertChannel(config.smtp_host, config.smtp_port, config.smtp_user, config.smtp_password, config.alert_email)
        if smtp_configured
        else ConsoleAlertChannel()
    )

    guard = AlertGuard(AlertGuardConfig(
        min_edge_threshold=config.min_edge_threshold,
        min_confidence_threshold=config.min_confidence_threshold,
        spam_window_hours=config.spam_window_hours,
    ))

    resolved = repo.get_all_finished("PL")
    baseline = (
        DixonColesStrategy().fit(resolved)
        if len(resolved) >= 20
        else EnhancedBaselineStrategy()
    )
    stats_agent = StatsMarketAgent(fixture_fetcher, odds_fetcher, baseline, FeatureExtractor(), repo)
    news_agent = NewsContextAgent(
        llm,
        ToolDispatcher(local_mcp_client),
        vector_repo,
        repo,
        repo,
        config.react_max_steps,
    )
    synthesis_agent = SynthesisAlertAgent(
        llm, alert_channel, repo, repo, guard,
        tuning=SynthesisTuning(base_sensitivity=config.base_sensitivity),
    )

    supervisor = SupervisorAgent(
        stats_agent, news_agent, synthesis_agent,
        min_edge_threshold=config.min_edge_threshold,
    )
    return supervisor.run(competition="PL", days_ahead=days_ahead)


def print_summary(forecasts: list, elapsed_seconds: float) -> None:
    """Print a concise run summary."""
    alerts = sum(1 for f in forecasts if f.alert_sent)
    print()
    print("=" * 60)
    print("  WEEKLY REFRESH SUMMARY")
    print("=" * 60)
    print(f"  Forecasts generated : {len(forecasts)}")
    print(f"  Alerts sent         : {alerts}")
    print(f"  No-alert decisions  : {len(forecasts) - alerts}")
    print(f"  Run time            : {elapsed_seconds:.1f}s")
    if forecasts:
        print()
        print(f"  {'Match':<40} {'Edge Market':<14} {'Edge':>7}  {'Conf':>6}  {'Alert'}")
        print(f"  {'-' * 40} {'-' * 14} {'-' * 7}  {'-' * 6}  {'-' * 5}")
        for f in forecasts:
            # Retrieve match label from forecast (match_id only — no lookup needed for summary)
            edge = f"{(f.edge_value or 0):+.1%}" if f.edge_value is not None else "  n/a"
            conf = f"{f.confidence_score:.0%}"
            sent = "YES" if f.alert_sent else "no"
            market = f.edge_market or "n/a"
            print(f"  {f.match_id:<40} {market:<14} {edge:>7}  {conf:>6}  {sent}")
    print("=" * 60)
    print()


def main(days_back: int = 14, days_ahead: int = 7) -> None:
    """Run the full weekly refresh: ingest outcomes, then run the forecast pipeline."""
    config = Config.from_env()
    conn = sqlite3.connect(config.db_path)
    init_db(conn)
    seed_source_reliability(conn)
    repo = SQLiteRepository(conn)

    fixture_fetcher = FixtureFetcher(config.football_data_api_key)
    odds_fetcher = OddsFetcher(config.odds_api_key)
    search_tool = WebSearchTool(config.search_api_key)

    start = datetime.now(timezone.utc)

    step("Step 1 — Ingesting recently resolved matches")
    n = ingest_resolved_matches(fixture_fetcher, repo, days_back=days_back)
    print(f"  Upserted {n} resolved matches into SQLite.")

    step("Step 2 — Running forecast pipeline")
    forecasts = run_forecast_pipeline(
        config, repo, fixture_fetcher, odds_fetcher, search_tool, days_ahead=days_ahead
    )

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    print_summary(forecasts, elapsed)
    conn.close()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Weekly refresh: ingest outcomes + run forecasts.")
    parser.add_argument("--days-back", type=int, default=14, help="Days of recent results to ingest (default: 14)")
    parser.add_argument("--days-ahead", type=int, default=7, help="Upcoming fixture window to forecast (default: 7)")
    args = parser.parse_args()
    main(days_back=args.days_back, days_ahead=args.days_ahead)
