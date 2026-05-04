"""Helpers for loading and rendering YAML-backed prompts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from soccer_forecast_agent.models.evidence import EvidenceItem
from soccer_forecast_agent.models.match import BaselineForecast, Match


PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def _load_prompt_file(name: str) -> dict:
    """Load and cache a YAML prompt file by stem name."""
    path = PROMPTS_DIR / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt file {path} must contain a mapping at the top level.")
    return data


def render_news_context_research_messages(
    match: Match,
    baseline: BaselineForecast,
    seed_context: list[str],
    evidence: list[EvidenceItem],
) -> list[dict[str, str]]:
    """Render the YAML-backed research prompt into chat messages."""
    prompts = _load_prompt_file("news_context")
    section = prompts["research"]
    evidence_preview = "\n".join(
        f"- {item.source}: {item.summary} [{item.direction} / {item.applies_to_market}]"
        for item in evidence[:5]
    ) or "None yet."
    seeded_text = "\n".join(f"- {item}" for item in seed_context) or "No prior vector context found."
    return [
        {"role": "system", "content": section["system"].strip()},
        {
            "role": "user",
            "content": section["user"].format(
                home_team=match.home_team,
                away_team=match.away_team,
                competition=match.competition,
                kickoff_time=match.kickoff_time.isoformat(),
                home_win=f"{baseline.home_win:.3f}",
                draw=f"{baseline.draw:.3f}",
                away_win=f"{baseline.away_win:.3f}",
                over_2_5=f"{baseline.over_2_5:.3f}",
                under_2_5=f"{baseline.under_2_5:.3f}",
                seeded_text=seeded_text,
                evidence_preview=evidence_preview,
            ).strip(),
        },
    ]


def render_news_context_evidence_messages(
    match: Match,
    baseline: BaselineForecast,
    source_material: str,
    source_label: str,
    max_items: int,
) -> list[dict[str, str]]:
    """Render the YAML-backed evidence extraction prompt into chat messages."""
    prompts = _load_prompt_file("news_context")
    section = prompts["evidence_extraction"]
    return [
        {"role": "system", "content": section["system"].strip()},
        {
            "role": "user",
            "content": section["user"].format(
                max_items=max_items,
                home_team=match.home_team,
                away_team=match.away_team,
                home_win=f"{baseline.home_win:.3f}",
                draw=f"{baseline.draw:.3f}",
                away_win=f"{baseline.away_win:.3f}",
                over_2_5=f"{baseline.over_2_5:.3f}",
                under_2_5=f"{baseline.under_2_5:.3f}",
                source_label=source_label,
                source_material=source_material,
            ).strip(),
        },
    ]
