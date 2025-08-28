"""Quickstart demo for ModelSearch."""
from __future__ import annotations
from sklearn.model_selection import train_test_split

from glassbox import ModelSearch
from glassbox.core.search import Search
from glassbox.core.evaluator import SklearnEvaluator
from glassbox.logger import logger


from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier

# Generate a larger synthetic dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Update the search space and model to be more complex
search = Search("grid", {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15],
    "learning_rate": [0.01, 0.1, 0.2]
})
ms = ModelSearch(
    model=GradientBoostingClassifier(),
    search=search,
    evaluator=SklearnEvaluator(),
    verbose=False,
)

best_model = ms.search(X_train, y_train)
logger.log(f"Best score: {best_model.score(X_test, y_test)}")