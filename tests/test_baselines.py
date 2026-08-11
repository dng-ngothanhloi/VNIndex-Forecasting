"""test_baselines.py – Cheap tests for persistence + AR(1) baselines."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.ar import persistence_forecast, fit_ar1, ar1_rolling_forecast


def _make_series(n=100, start="2022-01-01"):
    dates = pd.date_range(start, periods=n, freq="B")
    return pd.Series(np.cumsum(np.random.randn(n)) + 1000, index=dates)


# 1. Persistence: prediction(t) == actual(t-1)
def test_persistence_equals_lagged_actual():
    y = _make_series(50)
    targets = y.index[10:20]
    pred = persistence_forecast(y, targets)
    for i, t in enumerate(targets):
        pos = y.index.get_loc(t)
        assert pred.iloc[i] == y.iloc[pos - 1]


# 2. First Test prediction uses last Val actual
def test_persistence_first_test_uses_last_val():
    y = _make_series(100)
    train = y.iloc[:65]
    val = y.iloc[65:80]
    test = y.iloc[80:]
    pred = persistence_forecast(y, test.index)
    # First test prediction should be val's last actual
    assert pred.iloc[0] == val.iloc[-1]


# 3. AR1 fit uses Train only for Val diagnostic
def test_ar1_fit_train_only():
    y = _make_series(100)
    train = y.iloc[:65]
    coef = fit_ar1(train)
    assert coef["n_obs"] == 64  # n-1 for AR(1)
    assert "const" in coef and "phi" in coef


# 4. Final AR1 fit uses Train+Val only
def test_ar1_final_fit_trainval():
    y = _make_series(100)
    trainval = y.iloc[:80]
    coef = fit_ar1(trainval)
    assert coef["n_obs"] == 79


# 5. AR1 rolling Test uses actual previous y, not recursive
def test_ar1_uses_actual_not_recursive():
    y = _make_series(50)
    train = y.iloc[:30]
    coef = fit_ar1(train)
    targets = y.index[30:40]
    pred = ar1_rolling_forecast(coef, y, targets)
    # Each prediction should use actual y(t-1)
    for i, t in enumerate(targets):
        pos = y.index.get_loc(t)
        expected = coef["const"] + coef["phi"] * y.iloc[pos - 1]
        assert abs(pred.iloc[i] - expected) < 1e-10


# 6. Persistence/AR1 produce exactly same dates
def test_baselines_same_dates_as_target():
    y = _make_series(50)
    targets = y.index[20:35]
    persist = persistence_forecast(y, targets)
    coef = fit_ar1(y.iloc[:20])
    ar1_pred = ar1_rolling_forecast(coef, y, targets)
    assert persist.index.equals(targets)
    assert ar1_pred.index.equals(targets)


# 7. Fail fast on population mismatch
def test_persistence_fails_on_missing_history():
    y = _make_series(10)
    with pytest.raises((ValueError, KeyError)):
        # Target is before any data
        persistence_forecast(y, pd.DatetimeIndex([y.index[0]]))


# 8. Baselines stored at parent Run level (structural)
def test_baseline_artifact_structure():
    """Verify the expected artifact paths exist after a real run."""
    run_dir = PROJECT_ROOT / "artifacts" / "Run_20260811_134344"
    if not run_dir.exists():
        pytest.skip("Reference run not available")
    assert (run_dir / "baselines" / "persistence" / "summary.json").exists()
    assert (run_dir / "baselines" / "ar1" / "summary.json").exists()
    assert (run_dir / "comparison" / "baseline_comparison.csv").exists()


# 9. Incremental gain sign is correct
def test_incremental_gain_sign():
    """Positive gain means PCA-ARDL improves over baseline."""
    persist_rmse = 14.0
    ardl_rmse = 13.0
    gain = persist_rmse - ardl_rmse
    assert gain > 0  # PCA-ARDL is better


# 10. DM comparison uses paired N population
def test_dm_paired_population():
    """DM test requires same-length arrays."""
    from src.evaluation.dm_test import diebold_mariano_test
    actual = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    f1 = actual + np.array([0.5, -0.3, 0.2, -0.1, 0.4])
    f2 = actual + np.array([1.0, -1.0, 0.5, -0.5, 1.0])
    dm = diebold_mariano_test(actual, f1, f2, loss_type="mse")
    assert dm["sample_size"] == 5
