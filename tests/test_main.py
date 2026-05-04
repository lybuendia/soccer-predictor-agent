from dataclasses import dataclass

from soccer_forecast_agent.main import LocalMCPToolClient


@dataclass
class FakeFixtureFetcher:
    calls: list

    def fetch_upcoming(self, competition: str, days_ahead: int):
        self.calls.append({"competition": competition, "days_ahead": days_ahead})
        return ["fixture"]


@dataclass
class FakeOddsFetcher:
    calls: list

    def fetch_odds(self, home_team: str, away_team: str):
        self.calls.append({"home_team": home_team, "away_team": away_team})
        return {"odds": "ok"}


@dataclass
class FakeSearchTool:
    calls: list

    def search(self, query: str, max_results: int = 5):
        self.calls.append({"query": query, "max_results": max_results})
        return [{"title": "result"}]


def test_local_mcp_tool_client_dispatches_search_news() -> None:
    fixture_fetcher = FakeFixtureFetcher(calls=[])
    odds_fetcher = FakeOddsFetcher(calls=[])
    search_tool = FakeSearchTool(calls=[])
    client = LocalMCPToolClient(fixture_fetcher, odds_fetcher, search_tool)

    result = client.call_tool("search_news", {"query": "Arsenal Chelsea injuries", "max_results": 2})

    assert result == [{"title": "result"}]
    assert search_tool.calls == [{"query": "Arsenal Chelsea injuries", "max_results": 2}]


def test_local_mcp_tool_client_dispatches_odds_and_fixtures() -> None:
    fixture_fetcher = FakeFixtureFetcher(calls=[])
    odds_fetcher = FakeOddsFetcher(calls=[])
    search_tool = FakeSearchTool(calls=[])
    client = LocalMCPToolClient(fixture_fetcher, odds_fetcher, search_tool)

    fixtures = client.call_tool("get_fixtures", {"days_ahead": 5})
    odds = client.call_tool("get_odds", {"home_team": "Arsenal", "away_team": "Chelsea"})

    assert fixtures == ["fixture"]
    assert odds == {"odds": "ok"}
    assert fixture_fetcher.calls == [{"competition": "PL", "days_ahead": 5}]
    assert odds_fetcher.calls == [{"home_team": "Arsenal", "away_team": "Chelsea"}]
