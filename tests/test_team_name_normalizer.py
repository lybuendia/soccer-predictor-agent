from soccer_forecast_agent.domain.team_names import TeamNameNormalizer


def test_team_name_normalizer_maps_known_aliases_to_canonical_names():
    normalizer = TeamNameNormalizer()

    assert normalizer.canonicalize("Arsenal FC") == "Arsenal"
    assert normalizer.canonicalize("arsenal") == "Arsenal"
    assert normalizer.canonicalize("Man City") == "Manchester City"
    assert normalizer.canonicalize("Manchester United FC") == "Manchester United"
    assert normalizer.canonicalize("Tottenham Hotspur") == "Tottenham"
    assert normalizer.canonicalize("Brighton & Hove Albion") == "Brighton"


def test_team_name_normalizer_returns_trimmed_input_for_unknown_name():
    normalizer = TeamNameNormalizer()

    assert normalizer.canonicalize("Some New Club") == "Some New Club"
