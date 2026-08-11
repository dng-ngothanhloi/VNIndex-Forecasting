"""
ar.py – Persistence and AR(1) forecasting baselines
=====================================================
Minimal, clean implementations that follow the SAME protocol as the
corrected PCA-ARDL pipeline: rolling one-step-ahead, actual observed
history, T+1 forecast horizon, Train-only or Train+Val fit.

No PCA. No exogenous features. Uses only VNINDEX itself.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def persistence_forecast(y_full: pd.Series, target_dates: pd.DatetimeIndex) -> pd.Series:
    """Persistence baseline: y_hat(t) = actual y(t-1).
    
    Uses actual observed previous value (never a prior prediction).
    Requires y_full to span at least one observation before the first
    target_date through the last target_date.
    """
    preds = []
    for t in target_dates:
        # Find the position of t in y_full
        t_pos = y_full.index.get_loc(t)
        if t_pos == 0:
            raise ValueError(f"Cannot produce persistence forecast for {t}: no prior observation")
        y_prev = float(y_full.iloc[t_pos - 1])
        preds.append(y_prev)
    return pd.Series(preds, index=target_dates, name="Predicted_VNINDEX")


def fit_ar1(y_train: pd.Series) -> Dict[str, float]:
    """Fit AR(1): y_t = c + phi * y_{t-1} + eps using OLS.
    
    Returns dict with 'const', 'phi', 'n_obs'.
    """
    y = y_train.values[1:]      # y_t for t=1..n-1
    x = y_train.values[:-1]     # y_{t-1}
    n = len(y)

    # OLS: [c, phi] = (X'X)^-1 X'y where X = [1, y_{t-1}]
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    return {
        "const": float(beta[0]),
        "phi": float(beta[1]),
        "n_obs": n,
    }


def ar1_rolling_forecast(coef: Dict[str, float], y_full: pd.Series,
                         target_dates: pd.DatetimeIndex) -> pd.Series:
    """Fixed-coefficient AR(1) rolling one-step-ahead forecast.
    
    y_hat(t) = const + phi * actual_y(t-1)
    
    Uses ACTUAL observed y(t-1), never a prior prediction.
    Coefficients are frozen (no refitting per step).
    """
    c = coef["const"]
    phi = coef["phi"]
    preds = []
    for t in target_dates:
        t_pos = y_full.index.get_loc(t)
        if t_pos == 0:
            raise ValueError(f"Cannot produce AR(1) forecast for {t}: no prior observation")
        y_prev = float(y_full.iloc[t_pos - 1])
        y_hat = c + phi * y_prev
        preds.append(y_hat)
    return pd.Series(preds, index=target_dates, name="Predicted_VNINDEX")
