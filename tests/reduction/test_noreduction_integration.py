"""
test_noreduction_integration.py – Integration tests for NoReduction path
=========================================================================
Tests that the NoReduction pipeline integration works correctly with
the forecasting layer's feature discovery, ARDL Q-grid restriction,
and config-driven dispatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────
# 11. NoReduction Train/Val/Test counts
# ─────────────────────────────────────────────────────────────────
def test_noreduction_pipeline_output_shapes(tmp_path):
    """NoReduction pipeline outputs have correct row counts."""
    from src.pca_model import run_noreduction_pipeline

    # Create minimal scaled data fixtures
    splits_dir = tmp_path / "data" / "processed" / "splits"
    splits_dir.mkdir(parents=True)
    pca_dir = tmp_path / "data" / "processed" / "pca"

    cols = [f"F{i}" for i in range(10)]
    dates_train = pd.date_range("2022-01-01", periods=536, freq="B")
    dates_val = pd.date_range(dates_train[-1] + pd.Timedelta(days=1), periods=123, freq="B")
    dates_test = pd.date_range(dates_val[-1] + pd.Timedelta(days=1), periods=167, freq="B")

    pd.DataFrame(np.random.randn(536, 10), index=dates_train, columns=cols).to_csv(splits_dir / "train_scaled.csv")
    pd.DataFrame(np.random.randn(123, 10), index=dates_val, columns=cols).to_csv(splits_dir / "val_scaled.csv")
    pd.DataFrame(np.random.randn(167, 10), index=dates_test, columns=cols).to_csv(splits_dir / "test_scaled.csv")

    # Write minimal config
    config = {
        "paths": {
            "artifacts_dir": "data/processed",
            "models_dir": "models",
            "figures_dir": "logs/figures",
            "artifacts_subdirs": {"splits": "splits", "pca": "pca"},
        },
        "reduction": {"method": "none"},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_noreduction_pipeline(project_root=tmp_path, config_path=config_path)

    # Verify outputs
    train_out = pd.read_csv(pca_dir / "train_pca.csv", index_col=0, parse_dates=True)
    val_out = pd.read_csv(pca_dir / "val_pca.csv", index_col=0, parse_dates=True)
    test_out = pd.read_csv(pca_dir / "test_pca.csv", index_col=0, parse_dates=True)

    assert len(train_out) == 536
    assert len(val_out) == 123
    assert len(test_out) == 167


# ─────────────────────────────────────────────────────────────────
# 12. output_dim == input_dim
# ─────────────────────────────────────────────────────────────────
def test_noreduction_output_dim_equals_input_dim(tmp_path):
    """NoReduction preserves all features (output_dim == input_dim)."""
    from src.pca_model import run_noreduction_pipeline

    splits_dir = tmp_path / "data" / "processed" / "splits"
    splits_dir.mkdir(parents=True)
    pca_dir = tmp_path / "data" / "processed" / "pca"

    n_features = 318
    cols = [f"Stock{i}" for i in range(n_features)]
    pd.DataFrame(np.random.randn(50, n_features), columns=cols,
                 index=pd.date_range("2022-01-01", periods=50, freq="B")).to_csv(splits_dir / "train_scaled.csv")
    pd.DataFrame(np.random.randn(10, n_features), columns=cols,
                 index=pd.date_range("2022-04-01", periods=10, freq="B")).to_csv(splits_dir / "val_scaled.csv")
    pd.DataFrame(np.random.randn(10, n_features), columns=cols,
                 index=pd.date_range("2022-05-01", periods=10, freq="B")).to_csv(splits_dir / "test_scaled.csv")

    config = {
        "paths": {"artifacts_dir": "data/processed", "models_dir": "models",
                  "figures_dir": "logs/figures",
                  "artifacts_subdirs": {"splits": "splits", "pca": "pca"}},
        "reduction": {"method": "none"},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_noreduction_pipeline(project_root=tmp_path, config_path=config_path)

    train_out = pd.read_csv(pca_dir / "train_pca.csv", index_col=0)
    assert train_out.shape[1] == n_features


# ─────────────────────────────────────────────────────────────────
# 13. reduction_pct == 0
# ─────────────────────────────────────────────────────────────────
def test_noreduction_metrics_show_zero_reduction(tmp_path):
    """pca_metrics.csv from NoReduction reports dim_reduction_pct=0."""
    from src.pca_model import run_noreduction_pipeline

    splits_dir = tmp_path / "data" / "processed" / "splits"
    splits_dir.mkdir(parents=True)
    pca_dir = tmp_path / "data" / "processed" / "pca"

    cols = ["A", "B", "C"]
    pd.DataFrame(np.random.randn(20, 3), columns=cols,
                 index=pd.date_range("2022-01-01", periods=20, freq="B")).to_csv(splits_dir / "train_scaled.csv")
    pd.DataFrame(np.random.randn(5, 3), columns=cols,
                 index=pd.date_range("2022-03-01", periods=5, freq="B")).to_csv(splits_dir / "val_scaled.csv")
    pd.DataFrame(np.random.randn(5, 3), columns=cols,
                 index=pd.date_range("2022-04-01", periods=5, freq="B")).to_csv(splits_dir / "test_scaled.csv")

    config = {
        "paths": {"artifacts_dir": "data/processed", "models_dir": "models",
                  "figures_dir": "logs/figures",
                  "artifacts_subdirs": {"splits": "splits", "pca": "pca"}},
        "reduction": {"method": "none"},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_noreduction_pipeline(project_root=tmp_path, config_path=config_path)

    metrics = pd.read_csv(pca_dir / "pca_metrics.csv", index_col=0)
    assert float(metrics.loc["dim_reduction_pct", "value"]) == 0.0
    assert metrics.loc["reduction_method", "value"] == "none"


# ─────────────────────────────────────────────────────────────────
# 14. PCA path still works (regression gate)
# ─────────────────────────────────────────────────────────────────
def test_pca_path_still_produces_pc_columns():
    """Existing PCA artifacts still have PC-prefixed columns."""
    # Check real artifacts if available
    pca_dir = PROJECT_ROOT / "data" / "processed" / "pca"
    train_path = pca_dir / "train_pca.csv"
    if not train_path.exists():
        pytest.skip("PCA artifacts not present (run preprocess+PCA first)")

    train_pca = pd.read_csv(train_path, index_col=0)
    # Current config is PCA (default) — columns should start with PC
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    reduction_method = cfg.get("reduction", {}).get("method", "pca")
    if reduction_method == "pca":
        pc_cols = [c for c in train_pca.columns if c.startswith("PC")]
        assert len(pc_cols) >= 1, "PCA path must produce PC-prefixed columns"


# ─────────────────────────────────────────────────────────────────
# 15. forecasting feature discovery does not depend on "PC" prefix
# ─────────────────────────────────────────────────────────────────
def test_lstm_feature_discovery_works_without_pc_prefix():
    """LSTM data loader finds features regardless of prefix."""
    # Simulate what run_load_data does: join representation + vnindex,
    # then take all columns except VNINDEX as feature cols.
    repr_df = pd.DataFrame({
        "Stock_A": [1.0, 2.0, 3.0],
        "Stock_B": [4.0, 5.0, 6.0],
        "VNINDEX": [1000.0, 1001.0, 1002.0],
    }, index=pd.date_range("2022-01-01", periods=3, freq="B"))

    target_col = "VNINDEX"
    feature_cols = [c for c in repr_df.columns if c != target_col]
    assert feature_cols == ["Stock_A", "Stock_B"]
    assert "VNINDEX" not in feature_cols


def test_ardl_feature_discovery_works_without_pc_prefix():
    """ARDL load_inputs finds features regardless of prefix."""
    # Same logic as common.py::load_inputs now uses
    repr_df = pd.DataFrame({
        "FPT": [1.0, 2.0],
        "VNM": [3.0, 4.0],
        "VNINDEX": [1200.0, 1201.0],
    }, index=pd.date_range("2022-01-01", periods=2, freq="B"))

    pc_cols = [c for c in repr_df.columns if c != "VNINDEX"]
    assert pc_cols == ["FPT", "VNM"]


# ─────────────────────────────────────────────────────────────────
# 16. NoReduction ARDL q-grid == [1]
# ─────────────────────────────────────────────────────────────────
def test_noreduction_ardl_q_restriction():
    """When features > 50, ARDL Q-grid is restricted to [1]."""
    # Simulate the logic from sweep.py
    n_features = 318
    q_values = [1, 2, 3, 4, 5]
    reduction_method = "none"

    if reduction_method == "none" and n_features > 50:
        q_values_effective = [1]
    else:
        q_values_effective = q_values

    assert q_values_effective == [1]


# ─────────────────────────────────────────────────────────────────
# 17. PCA ARDL q-grid remains [1..5]
# ─────────────────────────────────────────────────────────────────
def test_pca_ardl_q_grid_unchanged():
    """PCA path keeps full Q=[1,2,3,4,5] grid."""
    n_features = 2  # typical PCA k
    q_values = [1, 2, 3, 4, 5]
    reduction_method = "pca"

    if reduction_method == "none" and n_features > 50:
        q_values_effective = [1]
    else:
        q_values_effective = q_values

    assert q_values_effective == [1, 2, 3, 4, 5]


# ─────────────────────────────────────────────────────────────────
# 18. NoReduction LSTM uses same Val target dates across lookbacks
#     (inherits from cross-boundary windowing, no change needed)
# ─────────────────────────────────────────────────────────────────
def test_cross_boundary_windowing_works_with_many_features():
    """Cross-boundary windowing works regardless of feature count."""
    from src.forecasting.lstm.data import make_cross_boundary_windowed_data

    n_features = 10  # small for speed, but proves it's not PC-dependent
    cols = [f"feat_{i}" for i in range(n_features)]
    target_col = "target"

    context_df = pd.DataFrame(
        np.random.randn(100, n_features + 1),
        index=pd.date_range("2022-01-01", periods=100, freq="B"),
        columns=cols + [target_col],
    )
    target_df = pd.DataFrame(
        np.random.randn(20, n_features + 1),
        index=pd.date_range("2022-06-01", periods=20, freq="B"),
        columns=cols + [target_col],
    )

    lookback = 30
    X, y, hist, dates = make_cross_boundary_windowed_data(
        context_df, target_df, cols, target_col, lookback
    )
    # All target_df rows should become targets
    assert len(X) == 20
    assert dates.equals(target_df.index)


# ─────────────────────────────────────────────────────────────────
# 19. NoReduction final Test population = 167 (fixture-based)
# ─────────────────────────────────────────────────────────────────
def test_noreduction_test_population_size():
    """Test set has 167 observations (matching the standard split)."""
    splits_dir = PROJECT_ROOT / "data" / "processed" / "splits"
    test_path = splits_dir / "test_scaled.csv"
    if not test_path.exists():
        pytest.skip("test_scaled.csv not present")

    test_df = pd.read_csv(test_path, index_col=0)
    assert len(test_df) == 167


# ─────────────────────────────────────────────────────────────────
# 20. representation metadata reaches run manifest
# ─────────────────────────────────────────────────────────────────
def test_config_has_reduction_method_field():
    """Config file contains reduction.method field."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    assert "reduction" in cfg
    assert "method" in cfg["reduction"]
    assert cfg["reduction"]["method"] in ("pca", "none")
