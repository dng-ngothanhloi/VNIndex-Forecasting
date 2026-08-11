"""
dm_test.py – Diebold-Mariano test for comparing forecast accuracy
=================================================================
Reference:
    Diebold, F.X. and Mariano, R.S. (1995).
    "Comparing Predictive Accuracy."
    Journal of Business & Economic Statistics, 13(3), 253-263.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Literal


def diebold_mariano_test(
    actual: np.ndarray,
    forecast1: np.ndarray,
    forecast2: np.ndarray,
    loss_type: Literal["mse", "mae"] = "mse",
    h: int = 1,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> dict:
    """
    Diebold-Mariano test for comparing predictive accuracy.

    H0: Equal predictive accuracy (E[d_t] = 0)
    H1: Forecast1 is better (d_bar < 0) or different (two-sided)

    Parameters
    ----------
    actual : array-like
        Realized values (test set).
    forecast1 : array-like
        Predictions from model 1 (e.g. ARDL).
    forecast2 : array-like
        Predictions from model 2 (e.g. LSTM).
    loss_type : {"mse", "mae"}
        Loss function for comparison.
    h : int
        Forecast horizon (1 = one-step ahead).
    alternative : {"two-sided", "greater", "less"}
        "less"    → H1: forecast1 better than forecast2 (d_bar < 0)
        "greater" → H1: forecast2 better than forecast1 (d_bar > 0)
        "two-sided" → H1: forecasts differ

    Returns
    -------
    dict with keys:
        dm_stat    : DM test statistic
        p_value    : p-value
        mean_diff  : mean loss differential d_bar
        se_diff    : HAC standard error
        sample_size: n
        max_lag    : lags used in HAC estimator
        conclusion : human-readable interpretation
        significant: True if p_value < 0.05
    """
    actual    = np.asarray(actual,    dtype=float)
    forecast1 = np.asarray(forecast1, dtype=float)
    forecast2 = np.asarray(forecast2, dtype=float)

    if not (len(actual) == len(forecast1) == len(forecast2)):
        raise ValueError("actual, forecast1, and forecast2 must have the same length")

    n = len(actual)
    e1 = actual - forecast1
    e2 = actual - forecast2

    # Loss differential
    if loss_type == "mse":
        d = e1 ** 2 - e2 ** 2
    elif loss_type == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"loss_type must be 'mse' or 'mae', got '{loss_type}'")

    d_bar = np.mean(d)

    # HAC variance estimator (Newey-West with Andrews 1991 lag selection)
    max_lag = max(1, min(int(np.floor(4 * (n / 100) ** (2 / 9))), n // 4))

    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for lag in range(1, max_lag + 1):
        cov_mat = np.cov(d[:-lag], d[lag:], ddof=1)
        gamma_k = cov_mat[0, 1]
        gamma_sum += (1.0 - lag / (max_lag + 1)) * gamma_k

    var_d = (gamma_0 + 2.0 * gamma_sum) / n
    se_d  = float(np.sqrt(max(var_d, 1e-12)))

    dm_stat = float(d_bar / se_d)

    # p-value under standard normal approximation
    if alternative == "two-sided":
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))
    elif alternative == "less":
        p_value = float(stats.norm.cdf(dm_stat))
    elif alternative == "greater":
        p_value = float(1.0 - stats.norm.cdf(dm_stat))
    else:
        raise ValueError(f"alternative must be 'two-sided', 'less', or 'greater'")

    # Significance label
    if p_value < 0.01:
        sig_label = "highly significant (p<0.01)"
    elif p_value < 0.05:
        sig_label = "significant (p<0.05)"
    elif p_value < 0.10:
        sig_label = "marginally significant (p<0.10)"
    else:
        sig_label = "not significant (p≥0.10)"

    if d_bar < 0:
        better = "forecast1"
    elif d_bar > 0:
        better = "forecast2"
    else:
        better = "equivalent"

    conclusion = (
        f"{better} performs better | {loss_type.upper()} diff={d_bar:.4f} | {sig_label}"
    )

    return {
        "dm_stat":     dm_stat,
        "p_value":     p_value,
        "mean_diff":   float(d_bar),
        "se_diff":     se_d,
        "sample_size": n,
        "max_lag":     max_lag,
        "conclusion":  conclusion,
        "significant": p_value < 0.05,
    }
