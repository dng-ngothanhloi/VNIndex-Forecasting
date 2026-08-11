"""Unit + property + Synthetic_Parity_Fixture + Real_Data_Smoke_Test tests
for the canonical ARDLForecaster (src/forecasting/ardl_forecaster.py).

Rewritten for the scientifically-corrected ARDL protocol (P0-2/P0-3A/P0-X,
forecasting-protocol-audit.md): causal=True, fixed hold_back, q>=1 (no
PC.L0), Train-only sweep fitting + rolling one-step-ahead Val evaluation,
ONE final Train+Val refit, rolling one-step-ahead Test forecast. This
replaces the pre-correction contract (optional X_val, q_values starting at
0, predict(X) without a target column) -- ARDLForecaster.fit() now
REQUIRES X_val/y_val and predict(X) now REQUIRES X to include the target
column (see class docstring for rationale).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.forecasting import ARDLForecaster
from src.forecasting.base import NotFittedError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_synthetic_ardl_data(rng: np.random.Generator, n_train=60, n_val=15, n_test=15, n_features=2):
    """Synthetic wide DataFrame with an autocorrelated target and PC-like
    exogenous features, shaped like the real train/val/test PCA+VNINDEX
    frames ARDLForecaster consumes."""
    n_total = n_train + n_val + n_test
    dates = pd.date_range("2020-01-01", periods=n_total, freq="B")
    X = pd.DataFrame(
        rng.normal(0.0, 1.0, size=(n_total, n_features)),
        index=dates,
        columns=[f"PC{i+1}" for i in range(n_features)],
    )
    y = np.zeros(n_total)
    y[0] = 100.0
    for t in range(1, n_total):
        y[t] = 0.9 * y[t - 1] + 0.5 * X.iloc[t, 0] + rng.normal(0, 0.1)
    y = pd.Series(y, index=dates, name="VNINDEX")

    X_train, X_val, X_test = X.iloc[:n_train], X.iloc[n_train:n_train + n_val], X.iloc[n_train + n_val:]
    y_train, y_val, y_test = y.iloc[:n_train], y.iloc[n_train:n_train + n_val], y.iloc[n_train + n_val:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def _predict_input(X: pd.DataFrame, y: pd.Series, target_col: str = "VNINDEX") -> pd.DataFrame:
    """predict(X) requires X to include the target column (rolling
    one-step-ahead forecasting needs actual observed history, P0-X)."""
    out = X.copy()
    out[target_col] = y.values
    return out


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_ardl_forecaster_public_api_is_minimal():
    f = ARDLForecaster()
    public_attrs = {name for name in dir(f) if not name.startswith("_")}
    assert {"fit", "predict", "get_metadata"}.issubset(public_attrs)


def test_predict_before_fit_raises_not_fitted_error():
    f = ARDLForecaster()
    rng = np.random.default_rng(0)
    X_train, y_train, X_val, y_val, X_test, y_test = make_synthetic_ardl_data(rng)
    with pytest.raises(NotFittedError):
        f.predict(_predict_input(X_test, y_test))


def test_get_metadata_before_fit_raises_not_fitted_error():
    f = ARDLForecaster()
    with pytest.raises(NotFittedError):
        f.get_metadata()


def test_fit_without_val_raises_value_error():
    """P0-3A requires a real Train->Val evaluation for candidate selection;
    fit() without X_val/y_val must raise, not silently fall back."""
    rng = np.random.default_rng(1)
    X_train, y_train, X_val, y_val, X_test, y_test = make_synthetic_ardl_data(rng, n_train=60, n_val=0, n_test=15)
    f = ARDLForecaster(p_values=[1, 2], q_values=[1])
    with pytest.raises(ValueError):
        f.fit(X_train, y_train)


def test_predict_without_target_column_raises_key_error():
    rng = np.random.default_rng(1)
    X_train, y_train, X_val, y_val, X_test, y_test = make_synthetic_ardl_data(rng, n_train=60, n_val=15, n_test=15)
    f = ARDLForecaster(p_values=[1, 2], q_values=[1])
    f.fit(X_train, y_train, X_val, y_val)
    with pytest.raises(KeyError):
        f.predict(X_test)  # missing target column


def test_value_error_propagates_when_no_model_succeeds():
    """Mirrors select_best_model's ValueError when no row has Status=='OK'."""
    rng = np.random.default_rng(2)
    X_train, y_train, X_val, y_val, X_test, y_test = make_synthetic_ardl_data(rng, n_train=20, n_val=5, n_test=5)
    f = ARDLForecaster(p_values=[999], q_values=[999])
    with pytest.raises(ValueError):
        f.fit(X_train, y_train, X_val, y_val)


def test_sweep_continues_past_per_pair_failures():
    rng = np.random.default_rng(3)
    X_train, y_train, X_val, y_val, X_test, y_test = make_synthetic_ardl_data(rng, n_train=60, n_val=15, n_test=15)
    f = ARDLForecaster(p_values=[1, 999], q_values=[1])
    f.fit(X_train, y_train, X_val, y_val)
    sweep_table = f.get_metadata()["sweep_table"]
    assert (sweep_table["Status"] == "OK").any()
    assert sweep_table["Status"].str.startswith("FAIL").any()


def test_use_ensemble_with_insufficient_candidates_silently_skips():
    rng = np.random.default_rng(4)
    X_train, y_train, X_val, y_val, X_test, y_test = make_synthetic_ardl_data(rng, n_train=60, n_val=15, n_test=15)
    f = ARDLForecaster(p_values=[1], q_values=[1], use_ensemble=True, ensemble_top_n=5)
    f.fit(X_train, y_train, X_val, y_val)  # only 1 candidate pair total
    meta = f.get_metadata()
    assert meta["ensemble_pairs"] is None
    preds = f.predict(_predict_input(X_test, y_test))
    assert len(preds) == len(X_test)


def test_get_metadata_contains_expected_keys():
    rng = np.random.default_rng(5)
    X_train, y_train, X_val, y_val, X_test, y_test = make_synthetic_ardl_data(rng)
    f = ARDLForecaster(p_values=[1, 2], q_values=[1, 2])
    f.fit(X_train, y_train, X_val, y_val)
    meta = f.get_metadata()
    assert set(meta.keys()) == {
        "method", "selected_pair", "causal", "hold_back", "metrics", "diagnostics",
        "sweep_table", "ensemble_pairs", "feature_columns",
    }
    assert meta["method"] == "ARDL"
    assert meta["causal"] is True


def test_default_q_values_exclude_zero_no_pc_l0():
    """P0-2: default q_values must not include 0 (no contemporaneous PC.L0)."""
    f = ARDLForecaster()
    assert 0 not in f.q_values
    assert f.causal is True


# ---------------------------------------------------------------------------
# Property tests (Hypothesis)
# ---------------------------------------------------------------------------

@settings(max_examples=15, deadline=None)
@given(n_test=st.integers(min_value=1, max_value=10))
def test_predict_returns_array_of_correct_length(n_test):
    rng = np.random.default_rng(42)
    X_train, y_train, X_val, y_val, X_test, y_test = make_synthetic_ardl_data(
        rng, n_train=60, n_val=15, n_test=n_test
    )
    f = ARDLForecaster(p_values=[1, 2], q_values=[1, 2])
    f.fit(X_train, y_train, X_val, y_val)
    preds = f.predict(_predict_input(X_test, y_test))
    assert len(preds) == n_test
    assert isinstance(preds, np.ndarray)


# ---------------------------------------------------------------------------
# Synthetic_Parity_Fixture: ARDLForecaster vs the underlying rolling helper
# ---------------------------------------------------------------------------

def test_ardl_forecaster_single_pair_matches_rolling_one_step_helper():
    """With a single (P, Q) candidate (no selection ambiguity), predict()
    must match a direct call to the shared rolling_one_step_forecast
    helper on the same final-refit ARDLResults, within tight tolerance."""
    from src.forecasting.ardl.common import rolling_one_step_forecast

    rng = np.random.default_rng(7)
    X_train, y_train, X_val, y_val, X_test, y_test = make_synthetic_ardl_data(rng, n_train=60, n_val=15, n_test=15)

    f = ARDLForecaster(p_values=[2], q_values=[1], selection_criterion="BIC", hold_back=2)
    f.fit(X_train, y_train, X_val, y_val)
    fw_preds = f.predict(_predict_input(X_test, y_test))

    y_trainval = pd.concat([y_train, y_val]).sort_index()
    X_trainval = pd.concat([X_train, X_val]).sort_index()
    y_full = pd.concat([y_trainval, y_test]).sort_index()
    X_full = pd.concat([X_trainval, X_test]).sort_index()
    ref_pred = rolling_one_step_forecast(f._final_res, y_full, X_full, y_test.index)

    np.testing.assert_allclose(fw_preds, ref_pred.values, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Real_Data_Smoke_Test: ARDLForecaster vs the canonical src/forecasting/ardl/* pipeline
# ---------------------------------------------------------------------------

def _real_data_available() -> bool:
    pca_dir = PROJECT_ROOT / "data" / "processed" / "pca"
    core_dir = PROJECT_ROOT / "data" / "processed" / "core"
    return all(
        (pca_dir / f).exists() for f in ["train_pca.csv", "val_pca.csv", "test_pca.csv"]
    ) and (core_dir / "vnindex_target.csv").exists()


@pytest.mark.skipif(not _real_data_available(), reason="real PCA/VNINDEX data not present in this environment")
def test_ardl_forecaster_matches_canonical_pipeline_on_real_data():
    """Full parity gate against the canonical src/forecasting/ardl/*
    pipeline on real data: selected pair (EXACT), sweep table grid/status
    (EXACT), predict() output and trainval metrics/diagnostics
    (rtol=1e-10, atol=1e-12)."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.forecasting.ardl.setup import run_setup, run_find_project_root
    from src.forecasting.ardl.data import run_load_data, run_validate_pca
    from src.forecasting.ardl.sweep import run_sweep_ardl
    from src.forecasting.ardl.forecast import run_select_and_forecast

    context = {}
    context = run_setup(context)
    context = run_find_project_root(context)
    context = run_load_data(context)
    context = run_validate_pca(context)
    context = run_sweep_ardl(context)
    context = run_select_and_forecast(context)

    f = ARDLForecaster(
        p_values=[1, 2, 3, 4, 5],
        q_values=[1, 2, 3, 4, 5],
        causal=True,
        hold_back=5,
        selection_criterion="BIC",
        criterion_threshold=4335.0,
        rmse_val_threshold=17.0,
    )
    X_train = context["train_df"][context["pc_cols"]]
    y_train = context["train_df"][context["target_col"]]
    X_val = context["val_df"][context["pc_cols"]]
    y_val = context["val_df"][context["target_col"]]
    X_test = context["test_df"][context["pc_cols"]]
    y_test = context["test_df"][context["target_col"]]

    f.fit(X_train, y_train, X_val, y_val)
    meta = f.get_metadata()

    assert meta["selected_pair"] == context["SELECTED_PAIR"]

    fw_sweep = meta["sweep_table"]
    lg_sweep = context["ardl_sweep_table"]
    assert fw_sweep[["P", "Q"]].equals(lg_sweep[["P", "Q"]])
    assert fw_sweep["Status"].equals(lg_sweep["Status"])
    for col in ["AIC", "BIC", "HQIC", "RMSE_val"]:
        np.testing.assert_allclose(fw_sweep[col].values, lg_sweep[col].values, rtol=1e-10, atol=1e-12, equal_nan=True)

    y_pred = f.predict(_predict_input(X_test, y_test))
    np.testing.assert_allclose(y_pred, context["pred_test"].values, rtol=1e-10, atol=1e-12)

    for k in ["RMSE_trainval", "MAE_trainval", "MAPE_trainval(%)"]:
        np.testing.assert_allclose(meta["metrics"][k], context["metrics"][k], rtol=1e-10, atol=1e-12)

    for k in context["diag"]:
        np.testing.assert_allclose(meta["diagnostics"][k], context["diag"][k], rtol=1e-10, atol=1e-12)
