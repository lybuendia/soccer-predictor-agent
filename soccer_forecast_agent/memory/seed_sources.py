"""Seeds the source_reliability table with default rule-based scores."""

import sqlite3
from datetime import datetime, timezone

DEFAULT_SOURCES = [
    ("premierleague.com",       "official",    0.95, "Official Premier League site"),
    ("bbc.co.uk",               "major_outlet", 0.85, "BBC Sport"),
    ("skysports.com",           "major_outlet", 0.85, "Sky Sports"),
    ("theguardian.com",         "major_outlet", 0.82, "The Guardian Sport"),
    ("espn.com",                "major_outlet", 0.80, "ESPN"),
    ("transfermarkt.com",       "data_site",   0.80, "Transfermarkt — injuries and squad data"),
    ("whoscored.com",           "data_site",   0.78, "WhoScored — stats"),
    ("fbref.com",               "data_site",   0.80, "FBref — advanced stats"),
    ("theathletic.com",         "major_outlet", 0.82, "The Athletic"),
    ("goal.com",                "major_outlet", 0.72, "Goal.com"),
    ("football365.com",         "secondary",   0.60, "Football365"),
    ("talksport.com",           "secondary",   0.58, "talkSPORT"),
    ("givemesport.com",         "secondary",   0.55, "GiveMeSport"),
    ("twitter.com",             "social",      0.35, "Twitter/X — treat as unverified"),
    ("reddit.com",              "social",      0.30, "Reddit — treat as unverified"),
]


def seed_source_reliability(connection: sqlite3.Connection) -> None:
    """Insert default source reliability scores; skip any sources already present."""
    now = datetime.now(timezone.utc).isoformat()
    connection.executemany(
        """
        INSERT OR IGNORE INTO source_reliability (source_id, source_name, category, reliability_score, notes, updated_at)
        VALUES (lower(hex(randomblob(8))), ?, ?, ?, ?, ?)
        """,
        [(name, cat, score, notes, now) for name, cat, score, notes in DEFAULT_SOURCES],
    )
    connection.commit()
