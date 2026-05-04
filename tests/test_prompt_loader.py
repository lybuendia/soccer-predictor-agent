from soccer_forecast_agent.models.evidence import EvidenceItem
from soccer_forecast_agent.prompts.loader import (
    render_news_context_evidence_messages,
    render_news_context_research_messages,
)


def test_render_news_context_research_messages_includes_dynamic_fields(sample_match, sample_baseline) -> None:
    evidence = [
        EvidenceItem(
            evidence_id="e1",
            forecast_id=sample_match.match_id,
            source="bbc.co.uk",
            url="https://bbc.co.uk/1",
            timestamp=sample_match.kickoff_time,
            summary="Home captain trained fully.",
            direction="home_positive",
            reliability_score=0.8,
            applies_to_market="winner",
        )
    ]

    messages = render_news_context_research_messages(
        match=sample_match,
        baseline=sample_baseline,
        seed_context=["bbc.co.uk | 2026-04-24 | teams=Arsenal | Arsenal striker trained."],
        evidence=evidence,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "NewsContextAgent" in messages[0]["content"]
    assert "Arsenal vs Chelsea" in messages[1]["content"]
    assert "home_win: 0.480" in messages[1]["content"]
    assert "Home captain trained fully." in messages[1]["content"]


def test_render_news_context_evidence_messages_includes_constraints(sample_match, sample_baseline) -> None:
    messages = render_news_context_evidence_messages(
        match=sample_match,
        baseline=sample_baseline,
        source_material="Fresh tool output about injuries and rotation risk.",
        source_label="search_news",
        max_items=2,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "JSON only" in messages[0]["content"]
    assert "Extract up to 2 soccer evidence items" in messages[1]["content"]
    assert "Allowed direction values: home_positive, away_positive, neutral, uncertainty." in messages[1]["content"]
    assert "Source label: search_news" in messages[1]["content"]
