import pytest

from soccer_forecast_agent.analytics.baseline import SimpleBaselineStrategy
from soccer_forecast_agent.analytics.features import FeatureExtractor


def test_simple_baseline_returns_normalized_probabilities(sample_match, sample_odds):
    context = FeatureExtractor().extract(
        match=sample_match,
        odds=sample_odds,
        home_results=["W", "W", "D", "W", "W"],
        away_results=["L", "D", "L", "W", "L"],
        home_goals_scored=[2, 3, 2, 1, 2],
        home_goals_conceded=[0, 1, 0, 1, 1],
        away_goals_scored=[1, 0, 1, 1, 0],
        away_goals_conceded=[2, 1, 2, 1, 3],
    )

    forecast = SimpleBaselineStrategy().compute(context)

    assert forecast.home_win + forecast.draw + forecast.away_win == pytest.approx(1.0, abs=1e-4)
    assert forecast.over_2_5 + forecast.under_2_5 == pytest.approx(1.0, abs=1e-4)
    assert 0.0 <= forecast.home_win <= 1.0
    assert 0.0 <= forecast.draw <= 1.0
    assert 0.0 <= forecast.away_win <= 1.0
    assert 0.0 <= forecast.over_2_5 <= 1.0
    assert 0.0 <= forecast.under_2_5 <= 1.0


def test_simple_baseline_favors_stronger_home_side(sample_match, sample_odds):
    context = FeatureExtractor().extract(
        match=sample_match,
        odds=sample_odds,
        home_results=["W", "W", "W", "D", "W"],
        away_results=["L", "L", "D", "L", "L"],
        home_goals_scored=[3, 2, 3, 2, 2],
        home_goals_conceded=[0, 1, 0, 1, 0],
        away_goals_scored=[0, 1, 0, 1, 0],
        away_goals_conceded=[2, 2, 3, 1, 2],
    )

    forecast = SimpleBaselineStrategy().compute(context)

    assert forecast.home_win > forecast.away_win
    assert forecast.home_win > forecast.draw


def test_simple_baseline_keeps_nonzero_away_win_probability_for_weak_side(sample_match, sample_odds):
    context = FeatureExtractor().extract(
        match=sample_match,
        odds=sample_odds,
        home_results=["W", "W", "W", "W", "W"],
        away_results=["L", "L", "L", "L", "L"],
        home_goals_scored=[3, 2, 3, 2, 3],
        home_goals_conceded=[0, 0, 1, 0, 1],
        away_goals_scored=[0, 0, 0, 0, 0],
        away_goals_conceded=[3, 2, 3, 2, 3],
    )

    forecast = SimpleBaselineStrategy().compute(context)

    assert forecast.away_win >= 0.05
    assert forecast.home_win + forecast.draw + forecast.away_win == pytest.approx(1.0, abs=1e-4)
