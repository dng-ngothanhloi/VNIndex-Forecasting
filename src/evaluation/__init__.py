"""src/evaluation – DM test, regression metrics, and CEV-level comparison
services (canonical, migrated from helpers/dm_test.py and compare/*.py)."""

from .dm_test import diebold_mariano_test
from .metrics import mae, mape, r2, regression_metrics, rmse
from .run_dm_test import load_forecasts, run_dm_test

__all__ = [
    "diebold_mariano_test",
    "regression_metrics",
    "rmse",
    "mae",
    "mape",
    "r2",
    "run_dm_test",
    "load_forecasts",
]
