"""Example usage of ModelTuner with XGBoost."""
from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from glassbox import ModelTuner


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

tuner = ModelTuner(
    model=XGBClassifier(tree_method="gpu_hist"),
    strategy="optuna",
    tracking="wandb",
    dashboard=True,
    enable_gpu=True,
)

best_model = tuner.search(X_train, y_train, time_limit="10m")
print("Best score:", best_model.score(X_test, y_test))
