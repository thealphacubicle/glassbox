"""Quickstart demo for ModelSearch."""
from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from glassbox import ModelSearch
from glassbox.core.search import Search
from glassbox.core.evaluator import SklearnEvaluator
from glassbox.logger import logger


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

search = Search("grid", {"C": [0.1, 1.0]})
ms = ModelSearch(model=LogisticRegression(max_iter=200), search=search, evaluator=SklearnEvaluator())

best_model = ms.search(X_train, y_train)
logger.log(f"Best score: {best_model.score(X_test, y_test)}")
