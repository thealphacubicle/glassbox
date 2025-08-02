from glassbox.config.defaults import SEARCH_SPACES


def test_search_spaces_present():
    assert "XGBClassifier" in SEARCH_SPACES
    assert "LogisticRegression" in SEARCH_SPACES
    assert "learning_rate" in SEARCH_SPACES["XGBClassifier"]
    assert "C" in SEARCH_SPACES["LogisticRegression"]
