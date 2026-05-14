"""Helpers for loading and rendering YAML-backed prompts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from soccer_forecast_agent.models.evidence import EvidenceItem, InterpretedEvidence
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


def render_synthesis_rationale_messages(
    match: Match,
    baseline: BaselineForecast,
    adjusted: BaselineForecast,
    interpreted: list[InterpretedEvidence],
    edge_market: str | None,
    evidence_limit: int,
    sentence_limit: int,
) -> list[dict[str, str]]:
    """Render the YAML-backed synthesis rationale prompt into chat messages."""
    prompts = _load_prompt_file("synthesis_alert")
    section = prompts["synthesis_rationale"]
    evidence_lines = (
        "\n".join(
            f"- {item.summary} [{item.source}]"
            for item in interpreted[:evidence_limit]
        )
        or "- No evidence collected."
    )
    return [
        {"role": "system", "content": section["system"].strip()},
        {
            "role": "user",
            "content": section["user"].format(
                sentence_limit=sentence_limit,
                home_team=match.home_team,
                away_team=match.away_team,
                baseline_home_win=f"{baseline.home_win:.1%}",
                baseline_draw=f"{baseline.draw:.1%}",
                baseline_away_win=f"{baseline.away_win:.1%}",
                adjusted_home_win=f"{adjusted.home_win:.1%}",
                adjusted_draw=f"{adjusted.draw:.1%}",
                adjusted_away_win=f"{adjusted.away_win:.1%}",
                adjusted_over_2_5=f"{adjusted.over_2_5:.1%}",
                adjusted_under_2_5=f"{adjusted.under_2_5:.1%}",
                edge_market=edge_market or "none",
                evidence_lines=evidence_lines,
            ).strip(),
        },
    ]
