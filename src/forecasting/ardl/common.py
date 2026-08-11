# --- moved as-is from ardl/step/common.py (Phase 2A consolidation) ---
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera


# ─────────────────────────────────────────────────────────────────
# P0-X: FIXED-MODEL ROLLING ONE-STEP-AHEAD FORECAST HELPER
# ─────────────────────────────────────────────────────────────────
# Scientific rationale (forecasting-protocol-audit.md, P0-X):
# statsmodels' ARDLResults.predict(start, end, exog_oos=...) over a
# multi-row OOS range does NOT perform rolling one-step forecasting using
# actual observed history. Inspecting statsmodels 0.14.6 source
# (statsmodels/tsa/ardl/model.py, ARDL.predict) shows that once the
# requested horizon exceeds max_1step (== min AR/exog lag under
# causal=True), it recursively substitutes the model's OWN prior
# forecasts into the AR lag slots for every step beyond the first -- it
# has no other source for y, since the fitted model object only retains
# the training-sample endog. This was confirmed empirically: a synthetic
# ARDL(1, {x:1}) OOS multi-step predict() diverged from a fixed-coefficient
# manual one-step-ahead forecast (using actual intervening y) by a max abs
# error of ~6.3 on a toy series where noise dominates the AR signal.
#
# This helper instead performs a genuine FIXED-MODEL rolling one-step
# forecast: fitted coefficients (`res.params`) are never re-estimated, and
# every lag term is read from ACTUAL observed history (never from a
# previous prediction), for every target date.
def rolling_one_step_forecast(
    res,
    y_full: pd.Series,
    exog_full: pd.DataFrame,
    target_dates: pd.Index,
) -> pd.Series:
    """Fixed-coefficient rolling one-step-ahead ARDL forecast.

    For every date ``t`` in ``target_dates``, predicts y(t) using the
    already-fitted ``res.params`` and ACTUAL endog/exog values at lags
    1..max_lag relative to t's row position in ``y_full``/``exog_full``
    (i.e. strictly information available through t-1). Never substitutes
    a previously *predicted* value and never reads PC(t)/y(t) (lag 0).

    Parameters
    ----------
    res : ARDLResults
        Already-fitted (Train-only, or Train+Val for the final refit)
        ARDL results object. Coefficients are read from ``res.params`` and
        held fixed -- this function does not refit anything.
    y_full : pd.Series
        ACTUAL target values, chronologically sorted, no duplicate dates,
        covering at least ``max_lag`` rows before the first
        ``target_dates`` entry through the last one (i.e. spans the
        preceding split's tail + the forecast split).
    exog_full : pd.DataFrame
        ACTUAL exogenous (PC) values aligned 1:1 with ``y_full`` (same
        index), covering the same range.
    target_dates : pd.Index
        The dates to forecast (e.g. all Val dates, or all Test dates).
        Must all be present in ``y_full.index``.

    Returns
    -------
    pd.Series
        Rolling one-step-ahead predictions, indexed by ``target_dates``.
    """
    model = res.model
    ar_lags = list(model._lags) if model._lags else []
    order: Dict = model._order or {}
    params = res.params

    if not y_full.index.is_monotonic_increasing:
        raise ValueError("y_full must be sorted ascending by date (chronological).")
    if y_full.index.has_duplicates:
        raise ValueError("y_full has duplicate dates; rolling one-step forecast requires a unique row per date.")
    if not exog_full.index.equals(y_full.index):
        raise ValueError("exog_full must share the exact same index as y_full.")

    y_name = getattr(model, "endog_names", None) or "y"
    const_val = float(params.get("const", 0.0))

    max_lag = max(ar_lags) if ar_lags else 0
    if order:
        max_lag = max(max_lag, max(max(v) for v in order.values() if v))

    missing_dates = [t for t in target_dates if t not in y_full.index]
    if missing_dates:
        raise ValueError(f"target_dates not found in y_full.index: {missing_dates[:5]} ...")

    preds = []
    for t in target_dates:
        pos = y_full.index.get_loc(t)
        if pos < max_lag:
            raise ValueError(
                f"Not enough history before {t} (position {pos}) to cover "
                f"max_lag={max_lag}. y_full must include the preceding "
                f"split's tail."
            )
        pred = const_val
        for lag in ar_lags:
            pred += float(params[f"{y_name}.L{lag}"]) * float(y_full.iloc[pos - lag])
        for col, lags in order.items():
            if not lags:
                continue
            for lag in lags:
                pred += float(params[f"{col}.L{lag}"]) * float(exog_full[col].iloc[pos - lag])
        preds.append(pred)

    return pd.Series(preds, index=pd.Index(target_dates), name="rolling_one_step_pred")


def find_project_root() -> Path:
    expected_rel = Path("data/processed/pca/train_pca.csv")
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
        # src/forecasting/ardl/common.py -> parents[3] is the project root
        # (parents[0]=ardl, [1]=forecasting, [2]=src, [3]=project root).
        Path(__file__).resolve().parents[3],
    ]

    for root in candidates:
        if (root / expected_rel).exists():
            return root

    if Path("/content").exists():
        for p in Path("/content").rglob("train_pca.csv"):
            if p.parent.name == "pca":
                root = p.parents[3]
                if (root / expected_rel).exists():
                    return root

    raise FileNotFoundError(
        "Cannot find data/processed/pca/train_pca.csv. Check the project folder location."
    )


def paired_valid(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


def diagnostics_from_residuals(resid) -> Dict[str, float]:
    lb_df = acorr_ljungbox(resid, lags=[1, 10], return_df=True)
    jb_stat, jb_pvalue, skew, kurt = jarque_bera(resid)
    arch_stat, arch_pvalue, _, _ = het_arch(resid, nlags=5)

    return {
        "LjungBox_Q_L1": float(lb_df.loc[1, "lb_stat"]),
        "LjungBox_p_L1": float(lb_df.loc[1, "lb_pvalue"]),
        "LjungBox_Q_L10": float(lb_df.loc[10, "lb_stat"]),
        "LjungBox_p_L10": float(lb_df.loc[10, "lb_pvalue"]),
        "JarqueBera": float(jb_stat),
        "JB_pvalue": float(jb_pvalue),
        "Skew": float(skew),
        "Kurtosis": float(kurt),
        "ARCH_stat": float(arch_stat),
        "ARCH_pvalue": float(arch_pvalue),
    }


def load_inputs(project_root: Path):
    pca_dir = project_root / "data/processed/pca"
    core_dir = project_root / "data/processed/core"

    train_pca = pd.read_csv(pca_dir / "train_pca.csv", parse_dates=["Ngày"]).set_index("Ngày")
    val_pca = pd.read_csv(pca_dir / "val_pca.csv", parse_dates=["Ngày"]).set_index("Ngày")
    test_pca = pd.read_csv(pca_dir / "test_pca.csv", parse_dates=["Ngày"]).set_index("Ngày")
    vnindex = pd.read_csv(core_dir / "vnindex_target.csv", parse_dates=["Ngày"]).set_index("Ngày")

    train_df = train_pca.join(vnindex, how="inner")
    val_df = val_pca.join(vnindex, how="inner")
    test_df = test_pca.join(vnindex, how="inner")
    pc_cols = [c for c in train_df.columns if c.startswith("PC")]

    return {
        "pca_dir": pca_dir,
        "core_dir": core_dir,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "pc_cols": pc_cols,
        "target_col": "VNINDEX",
    }
