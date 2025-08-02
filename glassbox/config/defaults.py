"""Default hyperparameter search spaces."""

SEARCH_SPACES = {
    "XGBClassifier": {
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7],
    },
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
    },
}
