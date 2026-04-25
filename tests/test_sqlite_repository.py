import sqlite3
from datetime import datetime, timezone

from soccer_forecast_agent.memory.seed_sources import seed_source_reliability
from soccer_forecast_agent.memory.sqlite_repository import SQLiteRepository, init_db
from soccer_forecast_agent.models.evidence import EvidenceItem
from soccer_forecast_agent.models.match import Match


def test_sqlite_repository_round_trips_match_forecast_and_evidence(sample_match, sample_forecast):
    connection = sqlite3.connect(":memory:")
    init_db(connection)
    repo = SQLiteRepository(connection)

    repo.save_match(sample_match)
    upcoming = repo.get_upcoming("PL")

    assert len(upcoming) == 1
    assert upcoming[0].home_team == "Arsenal"

    repo.save_forecast(sample_forecast)
    latest = repo.get_latest("match-1")

    assert latest is not None
    assert latest.forecast_id == "forecast-1"
    assert latest.baseline.home_win == sample_forecast.baseline.home_win

    evidence = EvidenceItem(
        evidence_id="ev-1",
        forecast_id="forecast-1",
        source="bbc.co.uk",
        url="https://bbc.co.uk/story",
        timestamp=datetime(2026, 4, 25, 13, 0, tzinfo=timezone.utc),
        summary="Injury report favors the home side.",
        direction="home_positive",
        reliability_score=0.8,
        applies_to_market="winner",
    )
    repo.save_evidence(evidence)
    stored_evidence = repo.get_by_forecast("forecast-1")

    assert len(stored_evidence) == 1
    assert stored_evidence[0].summary == evidence.summary


def test_sqlite_repository_updates_match_status_and_uses_seeded_source_scores(sample_match):
    connection = sqlite3.connect(":memory:")
    init_db(connection)
    seed_source_reliability(connection)
    repo = SQLiteRepository(connection)

    repo.save_match(sample_match)
    repo.update_status("match-1", "resolved", "2-1")

    assert repo.get_upcoming("PL") == []
    assert repo.get_score("bbc.co.uk") == 0.85
    assert repo.get_score("unknown-source.test") == 0.5


def test_sqlite_repository_returns_recent_finished_matches_for_team():
    connection = sqlite3.connect(":memory:")
    init_db(connection)
    repo = SQLiteRepository(connection)

    matches = [
        ("m1", "Arsenal", "Chelsea", "2026-04-01T12:00:00+00:00", "2-1"),
        ("m2", "Liverpool", "Arsenal", "2026-04-08T12:00:00+00:00", "1-1"),
        ("m3", "Arsenal", "Brighton", "2026-04-15T12:00:00+00:00", "3-0"),
        ("m4", "Tottenham", "Arsenal", "2026-04-22T12:00:00+00:00", "0-2"),
    ]
    for match_id, home, away, kickoff, score in matches:
        repo.save_match(
            Match(
                match_id=match_id,
                competition="PL",
                home_team=home,
                away_team=away,
                kickoff_time=datetime.fromisoformat(kickoff),
                status="resolved",
                final_score=score,
            )
        )

    recent = repo.get_recent_finished("Arsenal", "PL", limit=3)

    assert [match.match_id for match in recent] == ["m4", "m3", "m2"]
