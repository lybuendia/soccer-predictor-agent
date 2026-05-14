"""Inspect the history -> features -> baseline pipeline for a chosen fixture."""

import argparse
import sqlite3
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from soccer_forecast_agent.analytics.baseline import EnhancedBaselineStrategy
from soccer_forecast_agent.analytics.features import FeatureExtractor
from soccer_forecast_agent.agents.stats_market import StatsMarketAgent
from soccer_forecast_agent.config import Config
from soccer_forecast_agent.domain.team_names import TeamNameNormalizer
from soccer_forecast_agent.memory.sqlite_repository import SQLiteRepository
from soccer_forecast_agent.models.match import Match, MarketOdds


class _UnusedFixtureFetcher:
    def fetch_upcoming(self, competition: str, days_ahead: int) -> list[Match]:
        raise NotImplementedError


class _UnusedOddsFetcher:
    def fetch_odds(self, home_team: str, away_team: str):
        raise NotImplementedError


def main() -> None:
    """Print recent history, derived features, and baseline output for a fixture."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", default="Arsenal")
    parser.add_argument("--away", default="Chelsea")
    parser.add_argument("--competition", default="PL")
    args = parser.parse_args()

    load_dotenv()
    config = Config.from_env()
    connection = sqlite3.connect(config.db_path)
    repository = SQLiteRepository(connection)
    normalizer = TeamNameNormalizer()

    home_team = _resolve_team_name(connection, normalizer.canonicalize(args.home))
    away_team = _resolve_team_name(connection, normalizer.canonicalize(args.away))

    home_history = repository.get_recent_finished(home_team, args.competition, limit=5)
    away_history = repository.get_recent_finished(away_team, args.competition, limit=5)

    agent = StatsMarketAgent(
        fixture_fetcher=_UnusedFixtureFetcher(),
        odds_fetcher=_UnusedOddsFetcher(),
        baseline_strategy=EnhancedBaselineStrategy(home_advantage_boost=config.home_advantage_boost),
        feature_extractor=FeatureExtractor(),
        match_repo=repository,
    )

    match = Match(
        match_id=f"inspect-{home_team}-vs-{away_team}",
        competition=args.competition,
        home_team=home_team,
        away_team=away_team,
        kickoff_time=datetime.now(timezone.utc) + timedelta(days=1),
        status="upcoming",
    )
    odds = MarketOdds(
        odds_id="inspection-odds",
        match_id=match.match_id,
        timestamp=datetime.now(timezone.utc),
        home_win=2.0,
        draw=3.4,
        away_win=3.8,
        over_2_5=1.95,
        under_2_5=1.85,
    )

    home_results = agent._results_for_team(home_team, home_history)
    away_results = agent._results_for_team(away_team, away_history)
    home_goals_scored = agent._goals_scored_for_team(home_team, home_history)
    home_goals_conceded = agent._goals_conceded_for_team(home_team, home_history)
    away_goals_scored = agent._goals_scored_for_team(away_team, away_history)
    away_goals_conceded = agent._goals_conceded_for_team(away_team, away_history)

    context = FeatureExtractor().extract(
        match=match,
        odds=odds,
        home_results=home_results,
        away_results=away_results,
        home_goals_scored=home_goals_scored,
        home_goals_conceded=home_goals_conceded,
        away_goals_scored=away_goals_scored,
        away_goals_conceded=away_goals_conceded,
    )
    baseline = EnhancedBaselineStrategy(home_advantage_boost=config.home_advantage_boost).compute(context)

    print(f"Fixture: {home_team} vs {away_team}")
    print()
    _print_history("Home history", home_team, home_history, agent)
    print(f"Derived results: {home_results}")
    print(f"Goals scored: {home_goals_scored}")
    print(f"Goals conceded: {home_goals_conceded}")
    print()
    _print_history("Away history", away_team, away_history, agent)
    print(f"Derived results: {away_results}")
    print(f"Goals scored: {away_goals_scored}")
    print(f"Goals conceded: {away_goals_conceded}")
    print()
    print("Baseline forecast")
    print(f"  Home win : {baseline.home_win:.2%}")
    print(f"  Draw     : {baseline.draw:.2%}")
    print(f"  Away win : {baseline.away_win:.2%}")
    print(f"  Over 2.5 : {baseline.over_2_5:.2%}")
    print(f"  Under 2.5: {baseline.under_2_5:.2%}")


def _resolve_team_name(connection: sqlite3.Connection, team_query: str) -> str:
    """Resolve a user-provided team string to the canonical team name stored in SQLite."""
    names = {
        row[0]
        for row in connection.execute("SELECT DISTINCT home_team FROM matches").fetchall()
    } | {
        row[0]
        for row in connection.execute("SELECT DISTINCT away_team FROM matches").fetchall()
    }

    exact = next((name for name in names if name.casefold() == team_query.casefold()), None)
    if exact:
        return exact

    prefix = next((name for name in names if name.casefold().startswith(team_query.casefold())), None)
    if prefix:
        return prefix

    contains = next((name for name in names if team_query.casefold() in name.casefold()), None)
    if contains:
        return contains

    raise ValueError(f"No team found in SQLite matching '{team_query}'.")


def _print_history(title: str, team: str, matches: list[Match], agent: StatsMarketAgent) -> None:
    """Print recent finished match history for one team."""
    print(title)
    if not matches:
        print(f"  No recent finished matches found for {team}.")
        return
    for match in matches:
        result = agent._result_for_team(team, match)
        print(
            f"  {match.kickoff_time.date()} | {match.home_team} vs {match.away_team} | "
            f"{match.final_score} | result={result}"
        )


if __name__ == "__main__":
    main()
