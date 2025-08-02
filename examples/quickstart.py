"""Quickstart demo for ModelTuner."""
from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from glassbox import ModelTuner


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

tuner = ModelTuner(model=LogisticRegression(max_iter=200), strategy="grid")

best_model = tuner.search(X_train, y_train, time_limit="10s")
print("Best score:", best_model.score(X_test, y_test))
