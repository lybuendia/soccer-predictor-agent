"""SQLite implementation of all structured repository protocols."""

import sqlite3
from datetime import datetime
from pathlib import Path
from soccer_forecast_agent.models.match import Match, Forecast, BaselineForecast
from soccer_forecast_agent.models.evidence import EvidenceItem


def init_db(connection: sqlite3.Connection) -> None:
    """Run schema.sql against the connection to create all tables if they do not exist."""
    schema_path = Path(__file__).parent / "schema.sql"
    connection.executescript(schema_path.read_text())
    connection.commit()


class SQLiteRepository:
    """Implements MatchRepository, ForecastRepository, EvidenceRepository, and SourceReliabilityRepository against a SQLite connection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        """Initialise with an injected SQLite connection."""
        self._conn = connection
        self._conn.row_factory = sqlite3.Row

    # --- MatchRepository ---

    def save_match(self, match: Match) -> None:
        """Upsert a match record by match_id."""
        self._conn.execute(
            """
            INSERT INTO matches (match_id, competition, home_team, away_team, kickoff_time, status, final_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                competition=excluded.competition,
                home_team=excluded.home_team,
                away_team=excluded.away_team,
                kickoff_time=excluded.kickoff_time,
                status=excluded.status,
                final_score=excluded.final_score
            """,
            (
                match.match_id,
                match.competition,
                match.home_team,
                match.away_team,
                match.kickoff_time.isoformat(),
                match.status,
                match.final_score,
            ),
        )
        self._conn.commit()

    def get_upcoming(self, competition: str) -> list[Match]:
        """Return all upcoming matches for a competition ordered by kickoff_time."""
        rows = self._conn.execute(
            "SELECT * FROM matches WHERE competition = ? AND status = 'upcoming' ORDER BY kickoff_time",
            (competition,),
        ).fetchall()
        return [self._row_to_match(r) for r in rows]

    def get_recent_finished(
        self,
        team: str,
        competition: str,
        limit: int = 5,
        before: datetime | None = None,
    ) -> list[Match]:
        """Return the most recent finished matches for a team, optionally capped before a cutoff date."""
        query = """
            SELECT *
            FROM matches
            WHERE competition = ?
              AND status = 'resolved'
              AND final_score IS NOT NULL
              AND (home_team = ? OR away_team = ?)
        """
        params: list = [competition, team, team]
        if before is not None:
            query += "  AND kickoff_time < ?\n"
            params.append(before.isoformat())
        query += "ORDER BY kickoff_time DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_match(r) for r in rows]

    def get_all_finished(self, competition: str) -> list[Match]:
        """Return all resolved matches for a competition ordered chronologically."""
        rows = self._conn.execute(
            "SELECT * FROM matches WHERE competition = ? AND status = 'resolved' "
            "AND final_score IS NOT NULL ORDER BY kickoff_time ASC",
            (competition,),
        ).fetchall()
        return [self._row_to_match(r) for r in rows]

    def update_status(self, match_id: str, status: str, final_score: str | None = None) -> None:
        """Update match status and optionally record the final score."""
        self._conn.execute(
            "UPDATE matches SET status = ?, final_score = ? WHERE match_id = ?",
            (status, final_score, match_id),
        )
        self._conn.commit()

    # --- ForecastRepository ---

    def save_forecast(self, forecast: Forecast) -> None:
        """Persist a forecast record, including both baseline and adjusted probabilities."""
        b = forecast.baseline
        self._conn.execute(
            """
            INSERT INTO forecasts (
                forecast_id, match_id, run_timestamp,
                baseline_home_win, baseline_draw, baseline_away_win, baseline_over, baseline_under,
                adjusted_home_win, adjusted_draw, adjusted_away_win, adjusted_over, adjusted_under,
                confidence_score, edge_market, edge_value, alert_sent, rationale
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                forecast.forecast_id,
                forecast.match_id,
                forecast.run_timestamp.isoformat(),
                b.home_win, b.draw, b.away_win, b.over_2_5, b.under_2_5,
                forecast.adjusted_home_win,
                forecast.adjusted_draw,
                forecast.adjusted_away_win,
                forecast.adjusted_over_2_5,
                forecast.adjusted_under_2_5,
                forecast.confidence_score,
                forecast.edge_market,
                forecast.edge_value,
                int(forecast.alert_sent),
                forecast.rationale,
            ),
        )
        self._conn.commit()

    def get_by_match(self, match_id: str) -> list[Forecast]:
        """Return all forecasts for a match ordered by run_timestamp descending."""
        rows = self._conn.execute(
            "SELECT * FROM forecasts WHERE match_id = ? ORDER BY run_timestamp DESC",
            (match_id,),
        ).fetchall()
        return [self._row_to_forecast(r) for r in rows]

    def get_latest(self, match_id: str) -> Forecast | None:
        """Return the most recent forecast for a match, or None if none exists."""
        row = self._conn.execute(
            "SELECT * FROM forecasts WHERE match_id = ? ORDER BY run_timestamp DESC LIMIT 1",
            (match_id,),
        ).fetchone()
        return self._row_to_forecast(row) if row else None

    # --- EvidenceRepository ---

    def save_evidence(self, item: EvidenceItem) -> None:
        """Persist an evidence item."""
        self._conn.execute(
            """
            INSERT INTO evidence (evidence_id, forecast_id, source, url, timestamp, summary, direction, reliability_score, applies_to_market)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.evidence_id,
                item.forecast_id,
                item.source,
                item.url,
                item.timestamp.isoformat(),
                item.summary,
                item.direction,
                item.reliability_score,
                item.applies_to_market,
            ),
        )
        self._conn.commit()

    def get_by_forecast(self, forecast_id: str) -> list[EvidenceItem]:
        """Return all evidence items for a forecast."""
        rows = self._conn.execute(
            "SELECT * FROM evidence WHERE forecast_id = ?",
            (forecast_id,),
        ).fetchall()
        return [self._row_to_evidence(r) for r in rows]

    # --- SourceReliabilityRepository ---

    def get_score(self, source_domain: str) -> float:
        """Return the reliability score for a source domain; returns 0.5 if not found."""
        row = self._conn.execute(
            "SELECT reliability_score FROM source_reliability WHERE source_name = ?",
            (source_domain,),
        ).fetchone()
        return row[0] if row else 0.5

    # --- Private row mappers ---

    def _row_to_match(self, row: sqlite3.Row) -> Match:
        """Convert a sqlite3.Row to a Match dataclass."""
        return Match(
            match_id=row["match_id"],
            competition=row["competition"],
            home_team=row["home_team"],
            away_team=row["away_team"],
            kickoff_time=datetime.fromisoformat(row["kickoff_time"]),
            status=row["status"],
            final_score=row["final_score"],
        )

    def _row_to_forecast(self, row: sqlite3.Row) -> Forecast:
        """Convert a sqlite3.Row to a Forecast dataclass."""
        baseline = BaselineForecast(
            home_win=row["baseline_home_win"],
            draw=row["baseline_draw"],
            away_win=row["baseline_away_win"],
            over_2_5=row["baseline_over"],
            under_2_5=row["baseline_under"],
        )
        return Forecast(
            forecast_id=row["forecast_id"],
            match_id=row["match_id"],
            run_timestamp=datetime.fromisoformat(row["run_timestamp"]),
            baseline=baseline,
            adjusted_home_win=row["adjusted_home_win"],
            adjusted_draw=row["adjusted_draw"],
            adjusted_away_win=row["adjusted_away_win"],
            adjusted_over_2_5=row["adjusted_over"],
            adjusted_under_2_5=row["adjusted_under"],
            confidence_score=row["confidence_score"],
            edge_market=row["edge_market"],
            edge_value=row["edge_value"],
            alert_sent=bool(row["alert_sent"]),
            rationale=row["rationale"] or "",
        )

    def _row_to_evidence(self, row: sqlite3.Row) -> EvidenceItem:
        """Convert a sqlite3.Row to an EvidenceItem dataclass."""
        return EvidenceItem(
            evidence_id=row["evidence_id"],
            forecast_id=row["forecast_id"],
            source=row["source"],
            url=row["url"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            summary=row["summary"],
            direction=row["direction"],
            reliability_score=row["reliability_score"],
            applies_to_market=row["applies_to_market"],
        )
