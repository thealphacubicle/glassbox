# glassbox

**glassbox** is a developer-first, real-time observability layer for model tuning. It wraps existing hyperparameter search libraries and training pipelines with tracking and GPU awareness so you can see, understand and trust model experiments.

## Features
- Unified `ModelSearch` API for grid, random and Optuna-powered searches
- Optional Weights & Biases tracking
- Enhanced TQDM progress bars with spinner flair
- Unified `GlassboxLogger` routing messages to console and W&B
- Extensible plugin system with lifecycle hooks (e.g., Telegram notifications)
- GPU environment checks and model capability detection
- Lazy import helpers to keep dependencies optional
- Verbose mode to silence console logging unless explicitly enabled

## Installation
```bash
pip install glassbox
```

Optional extras can be installed as needed:
```bash
pip install glassbox[gpu]      # GPU libraries
pip install glassbox[wandb]    # Weights & Biases tracking
pip install glassbox[optuna]   # Optuna search backend
```

## Quick start
```python
from glassbox import ModelSearch
from glassbox.core.search import Search
from glassbox.core.evaluator import SklearnEvaluator
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

search = Search("grid", {"C": [0.1, 1.0]})
ms = ModelSearch(LogisticRegression(max_iter=100), search, SklearnEvaluator(), verbose=True)
model = ms.search(X_train, y_train)
print("accuracy", model.score(X_test, y_test))
```

See [examples/run_xgboost.py](glassbox/examples/run_xgboost.py) for an Optuna-based workflow with optional tracking and plugin support.

## Logging and Plugins

Use the global `logger` to route messages to different destinations and extend training with plugins:

```python
from glassbox.logger import logger
from glassbox.plugins.manager import PluginManager
from glassbox.plugins.knocknotifier import KnockNotifier
from glassbox.plugins import ResourceMonitor

plugin_manager = PluginManager()
plugin_manager.register(KnockNotifier(telegram_token="your-token", chat_id=123456))
plugin_manager.register(ResourceMonitor())

logger.log("Training started", to=["console", "wandb"])
plugin_manager.trigger("on_training_start")
```

Plugins listen to lifecycle hooks and should avoid blocking training. The included `KnockNotifier` sends a Telegram message when training completes if the `knockknock` package is installed.

## Testing
To run the project test suite:
```bash
pytest
```

## License
Distributed under the terms of the MIT license.
