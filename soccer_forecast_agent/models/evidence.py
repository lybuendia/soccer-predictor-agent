"""Data models for evidence items and ingested article chunks."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class EvidenceItem:
    """A single piece of qualitative evidence gathered by the ReAct research agent."""

    evidence_id: str
    forecast_id: str
    source: str
    url: str
    timestamp: datetime
    summary: str
    direction: str        # "home_positive" | "away_positive" | "neutral" | "uncertainty"
    reliability_score: float  # 0.0 – 1.0
    applies_to_market: str    # "winner" | "goals" | "both"


@dataclass
class InterpretedEvidence:
    """Per-market directional interpretation of an EvidenceItem, produced by the LLM at extraction time."""

    evidence_id: str
    source: str
    reliability_score: float
    winner_direction: str   # "home_positive" | "away_positive" | "draw_positive" | "neutral" | "uncertainty"
    goals_direction: str    # "over_positive" | "under_positive" | "neutral" | "uncertainty"
    market_weight: float    # 1.0 if evidence applies to both markets, 0.7 if single-market
    summary: str


@dataclass
class ArticleChunk:
    """A text chunk from an ingested article, stored in the vector store with team metadata."""

    chunk_id: str
    content: str
    source: str
    url: str
    published_at: str
    teams: list[str]
