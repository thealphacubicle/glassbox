from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

from glassbox.core.evaluator import evaluate


def test_evaluate_returns_float():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=50).fit(X, y)
    score = evaluate(model, X, y)
    assert isinstance(score, float)
    assert 0 <= score <= 1
