# Test Suite Overview

This directory contains unit tests for the glassbox MVP package.

| Test File | Purpose |
|-----------|---------|
| `test_lazy_imports.py` | Validates the optional import helper returns modules or raises informative errors. |
| `test_gpu.py` | Ensures GPU detection handles missing libraries and that model capability checks work. |
| `test_evaluator.py` | Confirms evaluation helper returns a valid score from scikit-learn models. |
| `test_search.py` | Exercises grid and random search strategies and verifies Optuna integration is optional. |
| `test_tuner.py` | Checks that the high-level `ModelTuner` orchestrates searches and enforces GPU guards. |
| `test_wandb_tracker.py` | Uses a dummy W&B client to verify tracking calls. |
| `test_dashboard.py` | Verifies the dashboard server requires the UI dependency. |
| `test_config.py` | Confirms default search spaces are populated for key models. |
| `test_logger.py` | Checks the unified logger routes messages to console and dashboard. |
| `test_plugins.py` | Ensures plugin hooks execute and the KnockNotifier handles missing dependencies. |
