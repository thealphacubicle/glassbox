"""Example usage of ModelSearch with XGBoost."""
from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from glassbox import ModelSearch
from glassbox.core.search import Search
from glassbox.core.evaluator import SklearnEvaluator
from glassbox.logger import logger


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

search = Search(
    "optuna",
    {"max_depth": [3, 5], "learning_rate": [0.1, 0.3]},
    n_trials=5,
)
ms = ModelSearch(
    model=XGBClassifier(tree_method="gpu_hist"),
    search=search,
    evaluator=SklearnEvaluator(),
    tracking="wandb",
    dashboard=True,
    enable_gpu=True,
)

best_model = ms.search(X_train, y_train)
logger.log(f"Best score: {best_model.score(X_test, y_test)}")
