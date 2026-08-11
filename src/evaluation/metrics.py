"""
metrics.py – Shared regression evaluation metrics (src/evaluation)
=======================================================================
Extracted from the inline `_standalone_metrics` helper previously defined
in compare/run_dm_test.py (P0-5 scientific correction), so both the DM
caller and any other evaluation code (e.g. multi-seed stability summaries,
Phase 3D) share one canonical RMSE/MAE/MAPE/R2 implementation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE / MAE / MAPE(%) / R2 for one (y_true, y_pred) pair."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE(%)": mape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
