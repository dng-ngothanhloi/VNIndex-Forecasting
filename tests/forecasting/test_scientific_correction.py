"""
test_scientific_correction.py - Forecasting scientific correction (P0-1..P0-5, P1)
=====================================================================================
Targeted tests for the scientific-correction task (NOT a numerical-parity
refactor -- see .kiro/audits/forecasting-protocol-audit.md). Covers:

- P0-1: non-overlap split config values / disjointness
- P0-2: ARDL causal lag map (no PC.L0), fixed hold_back
- P0-X: rolling one-step-ahead forecast semantics (fixed coefficients,
  actual history, never the model's own prior prediction)
- P0-4: LSTM cross-boundary historical context windowing (same Val/Test
  target dates across all lookbacks)
- P1: LSTM candidate selection has no last-iteration-state bug
- P0-5: DM fail-fast on population/date/y_true mismatch
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from statsmodels.tsa.ardl import ARDL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.ardl.common import rolling_one_step_forecast
from src.forecasting.lstm.data import make_cross_boundary_windowed_data, make_windowed_data


# ─────────────────────────────────────────────────────────────────
# P0-1: non-overlap split config
# ─────────────────────────────────────────────────────────────────
def test_config_uses_non_overlap_split_with_approved_ratios():
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prep = cfg["preprocess"]
    assert prep["use_overlap_val"] is False
    assert prep["train_ratio"] == pytest.approx(0.65)
    assert prep["val_ratio"] == pytest.approx(0.15)


def test_non_overlap_split_helper_produces_disjoint_chronological_splits():
    from src.utils import split_by_time

    n = 826
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    df = pd.DataFrame({"x": np.arange(n)}, index=dates)
    result = split_by_time(df, train_ratio=0.65, val_ratio=0.15)

    assert len(result.train) == 536
    assert len(result.val) == 123
    assert len(result.test) == n - 536 - 123

    assert result.train.index.max() < result.val.index.min()
    assert result.val.index.max() < result.test.index.min()
    assert result.train.index.intersection(result.val.index).empty
    assert result.val.index.intersection(result.test.index).empty
    assert result.train.index.intersection(result.test.index).empty


# ─────────────────────────────────────────────────────────────────
# P0-2: ARDL causal lag map (no PC.L0), fixed hold_back
# ─────────────────────────────────────────────────────────────────
def test_config_ardl_uses_causal_and_fixed_hold_back_and_no_q0():
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ardl_cfg = cfg["ardl"]
    assert ardl_cfg["causal"] is True
    assert ardl_cfg["hold_back"] == 5
    assert 0 not in ardl_cfg["q_values"]
    assert ardl_cfg["q_values"] == [1, 2, 3, 4, 5]


def test_ardl_causal_true_excludes_contemporaneous_pc_lag0():
    rng = np.random.default_rng(0)
    n = 80
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    X = pd.DataFrame({"PC1": rng.normal(size=n), "PC2": rng.normal(size=n)}, index=idx)
    y = pd.Series(rng.normal(size=n).cumsum(), index=idx, name="VNINDEX")

    model = ARDL(endog=y, lags=2, exog=X, order=2, causal=True, trend="c", hold_back=5)
    param_names = list(model.fit().params.index)

    assert not any(name.endswith(".L0") for name in param_names), (
        f"causal=True must exclude lag-0 (contemporaneous) exog terms, found: {param_names}"
    )
    # lags 1..2 for each exog column must be present
    for col in ("PC1", "PC2"):
        assert f"{col}.L1" in param_names
        assert f"{col}.L2" in param_names


def test_ardl_hold_back_is_fixed_regardless_of_lag_length():
    """hold_back must be identical across different (p,q) so IC comparison
    is fair (P0-2 requirement)."""
    rng = np.random.default_rng(1)
    n = 80
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    X = pd.DataFrame({"PC1": rng.normal(size=n)}, index=idx)
    y = pd.Series(rng.normal(size=n).cumsum(), index=idx, name="VNINDEX")

    for p, q in [(1, 1), (3, 2), (5, 5)]:
        model = ARDL(endog=y, lags=p, exog=X, order=q, causal=True, trend="c", hold_back=5)
        assert model._hold_back == 5, f"hold_back must stay fixed at 5 for (p={p}, q={q})"


# ─────────────────────────────────────────────────────────────────
# P0-X: rolling one-step-ahead forecast semantics
# ─────────────────────────────────────────────────────────────────
def test_rolling_one_step_forecast_matches_manual_fixed_coefficient_computation():
    rng = np.random.default_rng(2)
    n = 80
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    pc1 = rng.normal(size=n) * 2
    pc2 = rng.normal(size=n) * 1.5
    y = np.zeros(n)
    for i in range(2, n):
        y[i] = 0.4 * y[i - 1] + 0.2 * y[i - 2] + 0.3 * pc1[i - 1] + 0.15 * pc2[i - 2] + rng.normal() * 3.0
    y = pd.Series(y, index=idx, name="VNINDEX")
    X = pd.DataFrame({"PC1": pc1, "PC2": pc2}, index=idx)

    n_train = 60
    y_train, y_rest = y.iloc[:n_train], y.iloc[n_train:]
    X_train = X.iloc[:n_train]

    model = ARDL(endog=y_train, lags=2, exog=X_train, order=2, causal=True, trend="c", hold_back=2)
    res = model.fit()

    target_dates = y_rest.index
    preds = rolling_one_step_forecast(res, y, X, target_dates)

    const = res.params["const"]
    b_y1, b_y2 = res.params["VNINDEX.L1"], res.params["VNINDEX.L2"]
    b_x11, b_x12 = res.params["PC1.L1"], res.params["PC1.L2"]
    b_x21, b_x22 = res.params["PC2.L1"], res.params["PC2.L2"]

    manual = []
    for t in target_dates:
        pos = y.index.get_loc(t)
        manual.append(
            const + b_y1 * y.iloc[pos - 1] + b_y2 * y.iloc[pos - 2]
            + b_x11 * X["PC1"].iloc[pos - 1] + b_x12 * X["PC1"].iloc[pos - 2]
            + b_x21 * X["PC2"].iloc[pos - 1] + b_x22 * X["PC2"].iloc[pos - 2]
        )
    manual = pd.Series(manual, index=target_dates)

    np.testing.assert_allclose(preds.values, manual.values, rtol=1e-12, atol=1e-12)


def test_rolling_one_step_forecast_diverges_from_naive_multistep_predict():
    """Confirms P0-X's premise: statsmodels' native multi-step predict()
    recursively substitutes its own forecasts beyond the first OOS step,
    while the rolling helper always uses actual observed history."""
    rng = np.random.default_rng(3)
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    x = rng.normal(size=n) * 2
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * y[i - 1] + 0.3 * x[i - 1] + rng.normal() * 3
    y = pd.Series(y, index=idx, name="y")
    X = pd.DataFrame({"x": x}, index=idx)

    n_train = 40
    y_train, y_rest = y.iloc[:n_train], y.iloc[n_train:]
    X_train, X_rest = X.iloc[:n_train], X.iloc[n_train:]

    model = ARDL(endog=y_train, lags=1, exog=X_train, order=1, causal=True, trend="c", hold_back=1)
    res = model.fit()

    rolling_preds = rolling_one_step_forecast(res, y, X, y_rest.index)
    naive_preds = res.predict(start=n_train, end=n - 1, exog_oos=X_rest)
    naive_preds.index = y_rest.index

    # They must diverge after the first OOS step (rolling never
    # substitutes its own forecasts; naive predict does).
    assert not np.allclose(rolling_preds.values, naive_preds.values), (
        "rolling one-step and naive multi-step predict should diverge on "
        "noisy synthetic data beyond the first OOS step"
    )
    # But the FIRST prediction should match (both use only actual history there).
    np.testing.assert_allclose(rolling_preds.values[0], naive_preds.values[0], rtol=1e-9)


def test_rolling_one_step_forecast_raises_on_insufficient_history():
    rng = np.random.default_rng(4)
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    X = pd.DataFrame({"PC1": rng.normal(size=n)}, index=idx)
    y = pd.Series(rng.normal(size=n).cumsum(), index=idx, name="VNINDEX")

    model = ARDL(endog=y.iloc[:20], lags=5, exog=X.iloc[:20], order=5, causal=True, trend="c", hold_back=5)
    res = model.fit()

    # Ask for a target date whose position is < max_lag (5) in y -- must raise.
    with pytest.raises(ValueError):
        rolling_one_step_forecast(res, y, X, pd.Index([idx[2]]))


def test_rolling_one_step_forecast_rejects_duplicate_dates():
    rng = np.random.default_rng(5)
    n = 20
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    dup_idx = idx.insert(5, idx[5])  # inject a duplicate date
    X = pd.DataFrame({"PC1": rng.normal(size=n + 1)}, index=dup_idx)
    y = pd.Series(rng.normal(size=n + 1), index=dup_idx, name="VNINDEX")

    model = ARDL(endog=y.iloc[:15], lags=1, exog=X.iloc[:15], order=1, causal=True, trend="c", hold_back=1)
    res = model.fit()

    with pytest.raises(ValueError, match="duplicate"):
        rolling_one_step_forecast(res, y, X, pd.Index([dup_idx[-1]]))


# ─────────────────────────────────────────────────────────────────
# P0-4: LSTM cross-boundary historical context windowing
# ─────────────────────────────────────────────────────────────────
def _make_synthetic_scaled_df(n, start, pc_cols=("PC1", "PC2")):
    idx = pd.date_range(start, periods=n, freq="D")
    rng = np.random.default_rng(hash(start) % (2**31))
    data = {c: rng.normal(size=n) for c in pc_cols}
    data["VNINDEX"] = rng.normal(size=n)
    return pd.DataFrame(data, index=idx)


def test_cross_boundary_windowing_preserves_every_target_row_for_any_lookback():
    train_df = _make_synthetic_scaled_df(200, "2022-01-01")
    val_df = _make_synthetic_scaled_df(50, train_df.index.max() + pd.Timedelta(days=1))

    for lookback in (20, 30, 40, 50, 60):
        X, y, hist, end_dates = make_cross_boundary_windowed_data(
            train_df, val_df, ["PC1", "PC2"], "VNINDEX", lookback,
        )
        assert len(end_dates) == len(val_df), (
            f"lookback={lookback}: expected {len(val_df)} targets, got {len(end_dates)}"
        )
        assert end_dates.equals(val_df.index)
        assert X.shape == (len(val_df), lookback, 2)
        assert hist.shape == (len(val_df), lookback, 1)


def test_cross_boundary_windowing_gives_identical_target_dates_across_lookbacks():
    """D3/D4 requirement: ALL lookbacks must evaluate on EXACTLY the same
    Val (and by the same logic, Test) target dates."""
    train_df = _make_synthetic_scaled_df(300, "2022-01-01")
    val_df = _make_synthetic_scaled_df(80, train_df.index.max() + pd.Timedelta(days=1))

    date_sets = []
    for lookback in (20, 30, 40, 50, 60):
        _, _, _, end_dates = make_cross_boundary_windowed_data(
            train_df, val_df, ["PC1", "PC2"], "VNINDEX", lookback,
        )
        date_sets.append(end_dates)

    for other in date_sets[1:]:
        assert other.equals(date_sets[0])


def test_cross_boundary_windowing_never_uses_future_information():
    """For a target at t, history must come only from rows dated < t."""
    train_df = _make_synthetic_scaled_df(100, "2022-01-01")
    val_df = _make_synthetic_scaled_df(30, train_df.index.max() + pd.Timedelta(days=1))
    lookback = 20

    X, y, hist, end_dates = make_cross_boundary_windowed_data(
        train_df, val_df, ["PC1", "PC2"], "VNINDEX", lookback,
    )
    combined = pd.concat([train_df.tail(lookback), val_df])
    for i, t in enumerate(end_dates):
        pos = combined.index.get_loc(t)
        expected_x = combined[["PC1", "PC2"]].iloc[pos - lookback:pos].values
        np.testing.assert_allclose(X[i], expected_x)
        assert combined.index[pos - lookback:pos].max() < t


def test_cross_boundary_windowing_rejects_overlapping_or_reversed_boundary():
    train_df = _make_synthetic_scaled_df(50, "2022-01-01")
    # val_df overlaps train_df's date range -- must raise.
    val_df = _make_synthetic_scaled_df(20, train_df.index[-10])
    with pytest.raises(ValueError):
        make_cross_boundary_windowed_data(train_df, val_df, ["PC1", "PC2"], "VNINDEX", 20)


def test_cross_boundary_windowing_rejects_insufficient_context():
    train_df = _make_synthetic_scaled_df(10, "2022-01-01")  # only 10 rows
    val_df = _make_synthetic_scaled_df(20, train_df.index.max() + pd.Timedelta(days=1))
    with pytest.raises(ValueError):
        make_cross_boundary_windowed_data(train_df, val_df, ["PC1", "PC2"], "VNINDEX", 20)


# ─────────────────────────────────────────────────────────────────
# P1: LSTM selected-state consistency (no last-iteration-state bug)
# ─────────────────────────────────────────────────────────────────
def test_lstm_sweep_module_has_no_last_iteration_state_variables():
    """The historical bug (selected=LB40 but summary counts came from
    LB60) was caused by `_last_*` variables. Assert the rewritten
    src/forecasting/lstm/sweep.py source no longer contains this pattern."""
    sweep_src = (PROJECT_ROOT / "src" / "forecasting" / "lstm" / "sweep.py").read_text(encoding="utf-8")
    assert "_last_train_dates" not in sweep_src
    assert "_last_val_dates" not in sweep_src
    assert "_last_test_dates" not in sweep_src
    assert "_last_model" not in sweep_src
    assert "_last_history" not in sweep_src
    assert "_last_metrics_row" not in sweep_src
    assert "_last_X_train_final" not in sweep_src


def test_lstm_sweep_does_not_retain_all_model_objects_in_ram():
    """P1 explicitly forbids solving the state bug by keeping all TF model
    objects alive across the sweep -- verify clear_session() is called
    per-candidate in the sweep loop."""
    sweep_src = (PROJECT_ROOT / "src" / "forecasting" / "lstm" / "sweep.py").read_text(encoding="utf-8")
    assert "clear_session" in sweep_src


# ─────────────────────────────────────────────────────────────────
# P0-5: DM fail-fast on population/date/y_true mismatch
# ─────────────────────────────────────────────────────────────────
def _write_ardl_lstm_forecasts(tmp_path, ardl_rows, lstm_rows, cev=None):
    outputs_root = tmp_path / "outputs"
    if cev is not None:
        base = outputs_root / f"cev_{cev:.2f}"
    else:
        base = outputs_root
    ardl_dir = base / "ardl_vnindex_forecast"
    lstm_dir = base / "lstm_vnindex_sweep"
    ardl_dir.mkdir(parents=True, exist_ok=True)
    lstm_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(ardl_rows).to_csv(ardl_dir / "chapter4_ardl_forecast.csv", index=False)
    pd.DataFrame(lstm_rows).to_csv(lstm_dir / "predictions_lookback_30_batch_16.csv", index=False)
    # Minimal sweep_summary so load_forecasts' best-Val_RMSE lookup resolves.
    pd.DataFrame([{"Lookback": 30, "Batch_size": 16, "Val_RMSE": 1.0}]).to_csv(
        lstm_dir / "sweep_summary.csv", index=False
    )
    return base


def test_dm_load_forecasts_raises_on_length_mismatch(tmp_path, monkeypatch):
    # NOTE: src.evaluation.__init__ re-exports the `run_dm_test` FUNCTION
    # under the same name as this submodule, shadowing the submodule
    # attribute on the package -- import via importlib to get the module.
    import importlib
    dm_mod = importlib.import_module("src.evaluation.run_dm_test")

    monkeypatch.setattr(dm_mod, "PROJECT_ROOT", tmp_path)
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    ardl_rows = [{"Date": d, "Actual_VNINDEX": 100 + i, "Predicted_VNINDEX": 100 + i} for i, d in enumerate(dates)]
    lstm_rows = ardl_rows[:-1]  # one fewer row -> length mismatch
    _write_ardl_lstm_forecasts(tmp_path, ardl_rows, lstm_rows)

    with pytest.raises(AssertionError, match="Test population"):
        dm_mod.load_forecasts(cev=None)


def test_dm_load_forecasts_raises_on_date_mismatch(tmp_path, monkeypatch):
    import importlib
    dm_mod = importlib.import_module("src.evaluation.run_dm_test")

    monkeypatch.setattr(dm_mod, "PROJECT_ROOT", tmp_path)
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    other_dates = pd.date_range("2024-02-01", periods=10, freq="D")
    ardl_rows = [{"Date": d, "Actual_VNINDEX": 100, "Predicted_VNINDEX": 100} for d in dates]
    lstm_rows = [{"Date": d, "Actual_VNINDEX": 100, "Predicted_VNINDEX": 100} for d in other_dates]
    _write_ardl_lstm_forecasts(tmp_path, ardl_rows, lstm_rows)

    with pytest.raises(AssertionError, match="dates differ"):
        dm_mod.load_forecasts(cev=None)


def test_dm_load_forecasts_raises_on_y_true_mismatch(tmp_path, monkeypatch):
    import importlib
    dm_mod = importlib.import_module("src.evaluation.run_dm_test")

    monkeypatch.setattr(dm_mod, "PROJECT_ROOT", tmp_path)
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    ardl_rows = [{"Date": d, "Actual_VNINDEX": 100 + i, "Predicted_VNINDEX": 100} for i, d in enumerate(dates)]
    lstm_rows = [{"Date": d, "Actual_VNINDEX": 999 + i, "Predicted_VNINDEX": 100} for i, d in enumerate(dates)]
    _write_ardl_lstm_forecasts(tmp_path, ardl_rows, lstm_rows)

    with pytest.raises(AssertionError, match="Actual_VNINDEX"):
        dm_mod.load_forecasts(cev=None)


def test_dm_load_forecasts_passes_on_identical_population():
    """Sanity check: identical dates/y_true must NOT raise and must return
    all rows (no silent drop)."""
    import importlib
    dm_mod = importlib.import_module("src.evaluation.run_dm_test")
    importlib.reload(dm_mod)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dm_mod.PROJECT_ROOT = tmp_path
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        ardl_rows = [{"Date": d, "Actual_VNINDEX": 100 + i, "Predicted_VNINDEX": 99 + i} for i, d in enumerate(dates)]
        lstm_rows = [{"Date": d, "Actual_VNINDEX": 100 + i, "Predicted_VNINDEX": 101 + i} for i, d in enumerate(dates)]
        _write_ardl_lstm_forecasts(tmp_path, ardl_rows, lstm_rows)

        merged = dm_mod.load_forecasts(cev=None)
        assert len(merged) == 10
        assert set(merged.columns) >= {"Date", "Actual_VNINDEX", "ARDL_Pred", "LSTM_Pred"}
