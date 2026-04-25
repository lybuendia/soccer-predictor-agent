from soccer_forecast_agent.analytics.features import FeatureExtractor


def test_feature_extractor_builds_context_with_last_five_results(sample_match, sample_odds):
    extractor = FeatureExtractor()

    context = extractor.extract(
        match=sample_match,
        odds=sample_odds,
        home_results=["L", "D", "W", "W", "W", "D"],
        away_results=["W", "L", "L", "D", "L", "W"],
        home_goals_scored=[2, 1, 3, 2, 2, 4],
        home_goals_conceded=[0, 1, 1, 2, 0, 1],
        away_goals_scored=[1, 0, 1, 2, 1, 0],
        away_goals_conceded=[2, 2, 1, 1, 3, 2],
    )

    assert context.match == sample_match
    assert context.odds == sample_odds
    assert context.recent_home_results == ["D", "W", "W", "W", "D"]
    assert context.recent_away_results == ["L", "L", "D", "L", "W"]
    assert context.home_goals_scored_avg == 14 / 6
    assert context.home_goals_conceded_avg == 5 / 6
    assert context.away_goals_scored_avg == 5 / 6
    assert context.away_goals_conceded_avg == 11 / 6


def test_feature_extractor_uses_default_average_for_empty_series(sample_match, sample_odds):
    extractor = FeatureExtractor()

    context = extractor.extract(
        match=sample_match,
        odds=sample_odds,
        home_results=[],
        away_results=[],
        home_goals_scored=[],
        home_goals_conceded=[],
        away_goals_scored=[],
        away_goals_conceded=[],
    )

    assert context.home_goals_scored_avg == 1.2
    assert context.home_goals_conceded_avg == 1.2
    assert context.away_goals_scored_avg == 1.2
    assert context.away_goals_conceded_avg == 1.2
