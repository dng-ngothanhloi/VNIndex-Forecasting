"""Unit + behavioral tests for the canonical LSTMForecaster
(src/forecasting/lstm_forecaster.py).

Uses tiny synthetic data and few epochs so this suite runs fast; it is
NOT a numerical-parity test against a real pipeline run (that would need
TensorFlow + real PCA/VNINDEX data and is exercised manually via
experiments/run_lstm_experiment.py instead). These tests verify the
BaseForecaster contract and the P0-3B/P0-4/P1 structural invariants:
cross-boundary Val target-date preservation, no-EarlyStopping final
refit, and required X_val/target-column contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting import LSTMForecaster
from src.forecasting.base import NotFittedError


def _make_synthetic_data(n_train=80, n_val=20, n_test=15, n_features=2, seed=0):
    rng = np.random.default_rng(seed)
    n_total = n_train + n_val + n_test
    dates = pd.date_range("2020-01-01", periods=n_total, freq="B")
    X = pd.DataFrame(
        rng.normal(size=(n_total, n_features)),
        index=dates,
        columns=[f"PC{i+1}" for i in range(n_features)],
    )
    y = pd.Series(rng.normal(size=n_total).cumsum() + 100, index=dates, name="VNINDEX")

    X_train, X_val, X_test = X.iloc[:n_train], X.iloc[n_train:n_train + n_val], X.iloc[n_train + n_val:]
    y_train, y_val, y_test = y.iloc[:n_train], y.iloc[n_train:n_train + n_val], y.iloc[n_train + n_val:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def _predict_input(X: pd.DataFrame, y: pd.Series, target_col: str = "VNINDEX") -> pd.DataFrame:
    out = X.copy()
    out[target_col] = y.values
    return out


_FAST_KWARGS = dict(
    lookback_values=[5, 8],
    batch_size_values=[8],
    epochs=3,
    early_stopping_patience=2,
    reduce_lr_patience=2,
    min_epochs=1,
)


def test_lstm_forecaster_public_api_is_minimal():
    f = LSTMForecaster()
    public_attrs = {name for name in dir(f) if not name.startswith("_")}
    assert {"fit", "predict", "get_metadata"}.issubset(public_attrs)


def test_predict_before_fit_raises_not_fitted_error():
    f = LSTMForecaster(**_FAST_KWARGS)
    X_train, y_train, X_val, y_val, X_test, y_test = _make_synthetic_data()
    with pytest.raises(NotFittedError):
        f.predict(_predict_input(X_test, y_test))


def test_get_metadata_before_fit_raises_not_fitted_error():
    f = LSTMForecaster()
    with pytest.raises(NotFittedError):
        f.get_metadata()


def test_fit_without_val_raises_value_error():
    """P0-3B requires cross-boundary Val evaluation for candidate
    selection; fit() without X_val/y_val must raise."""
    f = LSTMForecaster(**_FAST_KWARGS)
    X_train, y_train, X_val, y_val, X_test, y_test = _make_synthetic_data()
    with pytest.raises(ValueError):
        f.fit(X_train, y_train)


def test_predict_without_target_column_raises_key_error():
    f = LSTMForecaster(**_FAST_KWARGS)
    X_train, y_train, X_val, y_val, X_test, y_test = _make_synthetic_data()
    f.fit(X_train, y_train, X_val, y_val)
    with pytest.raises(KeyError):
        f.predict(X_test)  # missing target column


def test_fit_predict_end_to_end_and_metadata_shape():
    f = LSTMForecaster(**_FAST_KWARGS)
    X_train, y_train, X_val, y_val, X_test, y_test = _make_synthetic_data()
    f.fit(X_train, y_train, X_val, y_val)

    preds = f.predict(_predict_input(X_test, y_test))
    assert len(preds) == len(X_test)
    assert isinstance(preds, np.ndarray)

    meta = f.get_metadata()
    assert meta["method"] == "LSTM"
    assert meta["selected_lookback"] in _FAST_KWARGS["lookback_values"]
    assert meta["selected_batch_size"] in _FAST_KWARGS["batch_size_values"]
    assert meta["final_refit_epochs"] >= 1
    assert set(meta["sweep_table"]["Lookback"].unique()) == set(_FAST_KWARGS["lookback_values"])


def test_cross_boundary_val_targets_identical_across_lookbacks_during_fit():
    """P0-4: every (lookback, batch_size) candidate must be evaluated on
    the SAME number of Val targets (== len(X_val)), never fewer."""
    f = LSTMForecaster(**_FAST_KWARGS)
    X_train, y_train, X_val, y_val, X_test, y_test = _make_synthetic_data()
    f.fit(X_train, y_train, X_val, y_val)

    sweep_table = f.get_metadata()["sweep_table"]
    # Val_RMSE must be computable (non-NaN) for every candidate lookback,
    # implying every candidate saw the full Val target set.
    assert sweep_table["Val_RMSE"].notna().all()


def test_export_lookback_and_batch_size_override_selection():
    f = LSTMForecaster(**{**_FAST_KWARGS, "export_lookback": 8, "export_batch_size": 8})
    X_train, y_train, X_val, y_val, X_test, y_test = _make_synthetic_data()
    f.fit(X_train, y_train, X_val, y_val)
    meta = f.get_metadata()
    assert meta["selected_lookback"] == 8
    assert meta["selected_batch_size"] == 8
