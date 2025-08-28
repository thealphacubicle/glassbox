"""Quickstart demo for ModelSearch."""
from __future__ import annotations
from sklearn.model_selection import train_test_split

from glassbox import ModelSearch
from glassbox.core.search import Search
from glassbox.core.evaluator import SklearnEvaluator
from glassbox.logger import logger


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Use a simpler dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simplify the search space and model
search = Search("grid", {
    "C": [0.1, 1.0, 10.0],
    "solver": ["lbfgs"],
    "penalty": ["l2"],
    "tol": [1e-4, 1e-3],
    "max_iter": [100, 200]
})
ms = ModelSearch(
    model=LogisticRegression(),
    search=search,
    evaluator=SklearnEvaluator(),
    verbose=False,
)

best_model = ms.search(X_train, y_train)
logger.log(f"Best score: {best_model.score(X_test, y_test)}")