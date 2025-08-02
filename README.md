# glassbox

**glassbox** is a developer-first, real-time observability layer for model tuning. It wraps existing hyperparameter search libraries and training pipelines with tracking, dashboards and GPU awareness so you can see, understand and trust model experiments.

## Features
- Unified `ModelTuner` API for grid, random and Optuna-powered searches
- Optional Weights & Biases tracking
- Lightweight dashboard powered by Streamlit/Reflex
- GPU environment checks and model capability detection
- Lazy import helpers to keep dependencies optional

## Installation
```bash
pip install glassbox
```

Optional extras can be installed as needed:
```bash
pip install glassbox[gpu]      # GPU libraries
pip install glassbox[wandb]    # Weights & Biases tracking
pip install glassbox[ui]       # Dashboard UI
pip install glassbox[optuna]   # Optuna search backend
```

## Quick start
```python
from glassbox import ModelTuner
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

mt = ModelTuner(LogisticRegression(max_iter=100), strategy="grid")
model = mt.search(X_train, y_train)
print("accuracy", model.score(X_test, y_test))
```

See [examples/run_xgboost.py](glassbox/examples/run_xgboost.py) for an Optuna-based workflow with optional tracking and dashboard support.

## Testing
To run the project test suite:
```bash
pytest
```

## License
Distributed under the terms of the MIT license.
