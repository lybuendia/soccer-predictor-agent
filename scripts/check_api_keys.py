"""Smoke-test all three external APIs and print a pass/fail summary."""

from dotenv import load_dotenv

load_dotenv()

from soccer_forecast_agent.config import Config
from soccer_forecast_agent.tools.fixtures import FixtureFetcher
from soccer_forecast_agent.tools.odds import OddsFetcher
from soccer_forecast_agent.tools.search import WebSearchTool


def _check(label: str, fn) -> bool:
    try:
        result = fn()
        print(f"  OK  {label}: {result}")
        return True
    except Exception as exc:
        print(f"  FAIL  {label}: {exc}")
        return False


def main() -> None:
    """Call each API with a minimal request and report pass/fail."""
    config = Config.from_env()
    results = []

    print("\n--- football-data.org (fixtures) ---")
    fetcher = FixtureFetcher(api_key=config.football_data_api_key)
    results.append(_check(
        "fetch upcoming PL fixtures (7 days)",
        lambda: f"{len(fetcher.fetch_upcoming('PL', days_ahead=7))} fixtures returned",
    ))

    print("\n--- the-odds-api.com (betting markets) ---")
    odds_fetcher = OddsFetcher(api_key=config.odds_api_key)
    results.append(_check(
        "fetch all upcoming EPL odds",
        lambda: f"{len(odds_fetcher.fetch_all_upcoming_odds())} markets returned",
    ))

    print("\n--- tavily.com (web search) ---")
    search = WebSearchTool(api_key=config.search_api_key)
    results.append(_check(
        "search 'Premier League team news'",
        lambda: f"{len(search.search('Premier League team news', max_results=3))} results returned",
    ))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} APIs reachable.")
    if passed < total:
        print("Check your .env keys for the failing APIs above.")


if __name__ == "__main__":
    main()
