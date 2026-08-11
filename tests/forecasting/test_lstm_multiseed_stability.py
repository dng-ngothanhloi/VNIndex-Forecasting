"""
test_lstm_multiseed_stability.py – Phase 3D multi-seed LSTM stability tests
================================================================================
Uses tiny synthetic data and few epochs for speed. Verifies:
- final_refit_and_forecast is reusable outside run_train_and_evaluate
  and reproducible for a fixed seed.
- Different seeds change results (weight init differs) but same-seed
  reruns are deterministic (TF determinism already enabled by
  src/forecasting/lstm/setup.py::run_imports at pipeline entry -- tested
  here via the model-building seed reset inside _build_lstm_model).
- Test date sets from different seeds are identical (same frozen
  hyperparameters / cross-boundary windowing, only weights differ).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.forecasting.lstm.sweep import final_refit_and_forecast


def _make_scaled_data(n_train=60, n_val=20, n_test=15, n_features=2, seed=0):
    rng = np.random.default_rng(seed)
    n_total = n_train + n_val + n_test
    dates = pd.date_range("2020-01-01", periods=n_total, freq="B")
    pc_cols = [f"PC{i+1}" for i in range(n_features)]
    target_col = "VNINDEX"

    X = rng.normal(size=(n_total, n_features))
    y = rng.normal(size=n_total).cumsum() + 100

    x_scaler = StandardScaler().fit(X[:n_train])
    y_scaler = StandardScaler().fit(y[:n_train].reshape(-1, 1))

    df = pd.DataFrame(x_scaler.transform(X), index=dates, columns=pc_cols)
    df[target_col] = y_scaler.transform(y.reshape(-1, 1)).ravel()

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    return train_df, val_df, test_df, pc_cols, target_col, y_scaler


_FAST_KWARGS = dict(
    selected_lookback=5,
    selected_batch_size=8,
    best_epoch=3,
    learning_rate=1e-3,
    lstm_units=[8, 4],
    dense_units=[4],
    dropout_rate=0.1,
    use_batch_norm=False,
)


def test_final_refit_and_forecast_returns_expected_shape():
    train_df, val_df, test_df, pc_cols, target_col, y_scaler = _make_scaled_data()
    result = final_refit_and_forecast(
        train_scaled_df=train_df, val_scaled_df=val_df, test_scaled_df=test_df,
        pc_cols=pc_cols, target_col=target_col, y_scaler=y_scaler,
        seed=42, **_FAST_KWARGS,
    )
    assert len(result["test_dates"]) == len(test_df)
    assert len(result["test_pred"]) == len(test_df)
    assert result["test_dates"].equals(test_df.index)
    assert "RMSE" not in result["metrics"]  # metrics keys are Test_RMSE/Dev_RMSE etc.
    assert "Test_RMSE" in result["metrics"]
    assert "Dev_RMSE" in result["metrics"]


def test_same_seed_produces_deterministic_predictions():
    """Two independent calls with the SAME seed and identical inputs must
    produce identical predictions (TF determinism via seed reset inside
    _build_lstm_model)."""
    train_df, val_df, test_df, pc_cols, target_col, y_scaler = _make_scaled_data()

    result1 = final_refit_and_forecast(
        train_scaled_df=train_df, val_scaled_df=val_df, test_scaled_df=test_df,
        pc_cols=pc_cols, target_col=target_col, y_scaler=y_scaler,
        seed=42, **_FAST_KWARGS,
    )
    result2 = final_refit_and_forecast(
        train_scaled_df=train_df, val_scaled_df=val_df, test_scaled_df=test_df,
        pc_cols=pc_cols, target_col=target_col, y_scaler=y_scaler,
        seed=42, **_FAST_KWARGS,
    )
    np.testing.assert_allclose(result1["test_pred"], result2["test_pred"], rtol=1e-5, atol=1e-5)


def test_different_seeds_produce_same_test_dates_but_different_predictions():
    """Different seeds must forecast the SAME Test target dates (frozen
    hyperparameters/windowing) but predictions may differ (different
    weight initialization is the only thing that should vary)."""
    train_df, val_df, test_df, pc_cols, target_col, y_scaler = _make_scaled_data()

    result_42 = final_refit_and_forecast(
        train_scaled_df=train_df, val_scaled_df=val_df, test_scaled_df=test_df,
        pc_cols=pc_cols, target_col=target_col, y_scaler=y_scaler,
        seed=42, **_FAST_KWARGS,
    )
    result_52 = final_refit_and_forecast(
        train_scaled_df=train_df, val_scaled_df=val_df, test_scaled_df=test_df,
        pc_cols=pc_cols, target_col=target_col, y_scaler=y_scaler,
        seed=52, **_FAST_KWARGS,
    )

    assert result_42["test_dates"].equals(result_52["test_dates"])
    assert result_42["y_test_true"].tolist() == pytest.approx(result_52["y_test_true"].tolist())
    # Predictions are very unlikely to be bit-identical across different
    # random initializations.
    assert not np.allclose(result_42["test_pred"], result_52["test_pred"], atol=1e-8)


def test_final_refit_never_uses_early_stopping_or_reduce_lr():
    """D4 invariant: the final refit must run for exactly best_epoch
    epochs with no EarlyStopping/ReduceLROnPlateau. Verified structurally
    by inspecting the shared helper's CODE BODY (excluding its docstring,
    which documents the invariant in prose) for callback usage."""
    import ast
    import inspect
    from src.forecasting.lstm import sweep as sweep_module

    source = inspect.getsource(sweep_module.final_refit_and_forecast)
    tree = ast.parse(source)
    func_node = tree.body[0]
    # Drop the docstring (first statement if it's an Expr/Constant str) so
    # prose mentions of "EarlyStopping" don't trip this structural check.
    body_without_docstring = func_node.body[1:] if (
        func_node.body and isinstance(func_node.body[0], ast.Expr)
    ) else func_node.body
    body_source = "\n".join(ast.unparse(n) for n in body_without_docstring)

    assert "EarlyStopping" not in body_source
    assert "ReduceLROnPlateau" not in body_source
    assert "callbacks" not in body_source
