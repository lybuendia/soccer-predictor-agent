-- SQLite schema for all structured memory tables.

CREATE TABLE IF NOT EXISTS matches (
    match_id     TEXT PRIMARY KEY,
    competition  TEXT NOT NULL,
    home_team    TEXT NOT NULL,
    away_team    TEXT NOT NULL,
    kickoff_time TEXT NOT NULL,
    status       TEXT NOT NULL,
    final_score  TEXT
);

CREATE TABLE IF NOT EXISTS market_odds (
    odds_id    TEXT PRIMARY KEY,
    match_id   TEXT NOT NULL REFERENCES matches(match_id),
    timestamp  TEXT NOT NULL,
    home_win   REAL NOT NULL,
    draw       REAL NOT NULL,
    away_win   REAL NOT NULL,
    over_2_5   REAL NOT NULL,
    under_2_5  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id        TEXT PRIMARY KEY,
    match_id           TEXT NOT NULL REFERENCES matches(match_id),
    run_timestamp      TEXT NOT NULL,
    baseline_home_win  REAL,
    baseline_draw      REAL,
    baseline_away_win  REAL,
    baseline_over      REAL,
    baseline_under     REAL,
    adjusted_home_win  REAL,
    adjusted_draw      REAL,
    adjusted_away_win  REAL,
    adjusted_over      REAL,
    adjusted_under     REAL,
    confidence_score   REAL,
    edge_market        TEXT,
    edge_value         REAL,
    alert_sent         INTEGER NOT NULL DEFAULT 0,
    rationale          TEXT
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id       TEXT PRIMARY KEY,
    forecast_id       TEXT NOT NULL REFERENCES forecasts(forecast_id),
    source            TEXT NOT NULL,
    url               TEXT,
    timestamp         TEXT NOT NULL,
    summary           TEXT NOT NULL,
    direction         TEXT NOT NULL,
    reliability_score REAL NOT NULL,
    applies_to_market TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_reliability (
    source_id         TEXT PRIMARY KEY,
    source_name       TEXT NOT NULL UNIQUE,
    category          TEXT NOT NULL,
    reliability_score REAL NOT NULL,
    notes             TEXT,
    updated_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_preferences (
    preference_id           TEXT PRIMARY KEY,
    preferred_competition   TEXT,
    preferred_markets       TEXT,
    min_edge_threshold      REAL,
    min_confidence_threshold REAL,
    lineup_rerun_enabled    INTEGER,
    email_address           TEXT
);
