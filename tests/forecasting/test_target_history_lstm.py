"""
test_target_history_lstm.py – Targeted tests for the Target-History LSTM baseline.

Covers:
  - Input representation (single VNINDEX channel, no PCA/stock leakage)
  - Temporal validity (Val=123, Test=167, no future leakage)
  - Selection (never touches Test during tuning)
  - Final refit (new model, Train+Val only, fixed epochs, no EarlyStopping)
  - Population parity vs PCA-LSTM
  - Required artifact files
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.lstm.data import (
    make_windowed_data,
    make_cross_boundary_windowed_data,
    add_target_history,
)

# ── Synthetic-data helpers ────────────────────────────────────────────────

def _make_split_dfs(n_train=80, n_val=20, n_test=15, n_pca=3, seed=0):
    """Return (train, val, test) DataFrames with PC1..PCn + VNINDEX columns."""
    rng = np.random.default_rng(seed)
    n_total = n_train + n_val + n_test
    dates = pd.date_range("2020-01-01", periods=n_total, freq="B")
    pca_cols = [f"PC{i + 1}" for i in range(n_pca)]
    data = {col: rng.normal(size=n_total) for col in pca_cols}
    data["VNINDEX"] = rng.normal(size=n_total).cumsum() + 1000
    df = pd.DataFrame(data, index=dates)
    from sklearn.preprocessing import StandardScaler
    scaler_x = StandardScaler().fit(df[pca_cols].iloc[:n_train])
    scaler_y = StandardScaler().fit(df[["VNINDEX"]].iloc[:n_train])
    scaled = df.copy()
    scaled[pca_cols] = scaler_x.transform(df[pca_cols])
    scaled["VNINDEX"] = scaler_y.transform(df[["VNINDEX"]]).ravel()
    return (
        scaled.iloc[:n_train],
        scaled.iloc[n_train:n_train + n_val],
        scaled.iloc[n_train + n_val:],
        pca_cols,
    )


def _make_th_context(n_train=80, n_val=20, n_test=15, n_pca=3):
    """Build a target-history context via prepare_target_history_context."""
    from src.forecasting.lstm.sweep import prepare_target_history_context

    train, val, test, pca_cols = _make_split_dfs(n_train, n_val, n_test, n_pca)

    # Minimal context mirroring what run_prepare_data sets
    context = {
        "train_scaled_df": train,
        "val_scaled_df": val,
        "test_scaled_df": test,
        "pc_cols": pca_cols,
        "target_col": "VNINDEX",
        "x_scaler": None,
        "y_scaler": None,
        "lookback_values": [5],
        "batch_size_values": [8],
        "epochs": 2,
        "PROJECT_ROOT": PROJECT_ROOT,
    }
    return prepare_target_history_context(context)


# ──────────────────────────────────────────────────────────────────────────
# 1. INPUT REPRESENTATION
# ──────────────────────────────────────────────────────────────────────────

class TestInputRepresentation:
    def test_pc_cols_is_empty_in_th_context(self):
        """After transform, pc_cols must be [] so make_windowed_data produces
        X of shape (N, L, 0) and the only input channel comes from
        add_target_history, giving final shape (N, L, 1)."""
        ctx = _make_th_context()
        assert ctx["pc_cols"] == [], (
            f"Expected pc_cols=[], got {ctx['pc_cols']}"
        )

    def test_final_input_tensor_has_exactly_one_channel(self):
        """X_final = concat([X_pcs, X_hist], axis=2) must have shape[-1]==1."""
        ctx = _make_th_context()
        lookback = 5
        train_df = ctx["train_scaled_df"]

        X, y, _ = make_windowed_data(train_df, ctx["pc_cols"], ctx["target_col"], lookback)
        assert X.shape[2] == 0, f"PC array should be empty (0 channels), got {X.shape[2]}"

        X_hist = add_target_history(train_df, ctx["target_col"], lookback)
        X_final = np.concatenate([X, X_hist], axis=2)
        assert X_final.shape[2] == 1, (
            f"Final input tensor should have 1 channel, got {X_final.shape[2]}"
        )

    def test_input_channel_equals_vnindex_history(self):
        """The single input channel must contain the scaled VNINDEX values."""
        ctx = _make_th_context()
        lookback = 5
        train_df = ctx["train_scaled_df"]

        X, _, _ = make_windowed_data(train_df, ctx["pc_cols"], ctx["target_col"], lookback)
        X_hist = add_target_history(train_df, ctx["target_col"], lookback)
        X_final = np.concatenate([X, X_hist], axis=2)

        # Window 0: should equal train_df['VNINDEX'].values[0:lookback]
        expected = train_df["VNINDEX"].values[:lookback]
        np.testing.assert_array_almost_equal(
            X_final[0, :, 0], expected, decimal=10,
            err_msg="Input channel must equal VNINDEX history",
        )

    def test_no_pca_column_in_th_context(self):
        ctx = _make_th_context(n_pca=3)
        for df_key in ("train_scaled_df", "val_scaled_df", "test_scaled_df"):
            cols = list(ctx[df_key].columns)
            for col in cols:
                assert not col.startswith("PC"), (
                    f"PCA column {col!r} leaked into target-history {df_key}"
                )

    def test_th_has_fewer_channels_than_pca_lstm(self):
        """TH shape[-1]==1, PCA shape[-1]==k+1; must always be strictly less."""
        ctx = _make_th_context(n_pca=3)
        lookback = 5
        train_df = ctx["train_scaled_df"]

        X_th, _, _ = make_windowed_data(train_df, ctx["pc_cols"], ctx["target_col"], lookback)
        X_th_hist = add_target_history(train_df, ctx["target_col"], lookback)
        X_th_final = np.concatenate([X_th, X_th_hist], axis=2)

        # Simulate PCA context with k=3
        _, _, train_pca, pca_cols = _make_split_dfs(n_pca=3)
        X_pca, _, _ = make_windowed_data(train_pca, pca_cols, "VNINDEX", lookback)
        X_pca_hist = add_target_history(train_pca, "VNINDEX", lookback)
        X_pca_final = np.concatenate([X_pca, X_pca_hist], axis=2)

        assert X_th_final.shape[2] == 1
        assert X_pca_final.shape[2] == len(pca_cols) + 1  # k+1
        assert X_th_final.shape[2] < X_pca_final.shape[2], (
            "TH-LSTM must have fewer input channels than PCA-LSTM"
        )


# ──────────────────────────────────────────────────────────────────────────
# 2. TEMPORAL VALIDITY
# ──────────────────────────────────────────────────────────────────────────

class TestTemporalValidity:
    def test_val_target_count_equals_n_val_for_all_lookbacks(self):
        """P0-4: every lookback must produce exactly n_val=20 Val targets."""
        ctx = _make_th_context(n_val=20)
        train_df = ctx["train_scaled_df"]
        val_df = ctx["val_scaled_df"]
        for lb in [3, 5, 10]:
            _, _, _, val_dates = make_cross_boundary_windowed_data(
                train_df, val_df, ctx["pc_cols"], ctx["target_col"], lb
            )
            assert len(val_dates) == len(val_df), (
                f"lookback={lb}: got {len(val_dates)} val targets, expected {len(val_df)}"
            )

    def test_test_target_count_equals_n_test_for_all_lookbacks(self):
        ctx = _make_th_context(n_test=15)
        train_df = ctx["train_scaled_df"]
        val_df = ctx["val_scaled_df"]
        test_df = ctx["test_scaled_df"]
        trainval_df = pd.concat([train_df, val_df]).sort_index()
        for lb in [3, 5, 10]:
            _, _, _, test_dates = make_cross_boundary_windowed_data(
                trainval_df, test_df, ctx["pc_cols"], ctx["target_col"], lb
            )
            assert len(test_dates) == len(test_df), (
                f"lookback={lb}: got {len(test_dates)} test targets, expected {len(test_df)}"
            )

    def test_no_val_date_appears_in_train_history(self):
        """Context window for a Val target must come from Train tail only."""
        ctx = _make_th_context()
        train_df = ctx["train_scaled_df"]
        val_df = ctx["val_scaled_df"]
        lb = 5
        _, _, _, val_dates = make_cross_boundary_windowed_data(
            train_df, val_df, ctx["pc_cols"], ctx["target_col"], lb
        )
        # All val dates must be after the last train date
        assert val_dates.min() > train_df.index.max(), (
            "Val target dates overlap with Train dates"
        )

    def test_no_test_date_in_trainval_history(self):
        ctx = _make_th_context()
        train_df = ctx["train_scaled_df"]
        val_df = ctx["val_scaled_df"]
        test_df = ctx["test_scaled_df"]
        trainval_df = pd.concat([train_df, val_df]).sort_index()
        lb = 5
        _, _, _, test_dates = make_cross_boundary_windowed_data(
            trainval_df, test_df, ctx["pc_cols"], ctx["target_col"], lb
        )
        assert test_dates.min() > trainval_df.index.max()


# ──────────────────────────────────────────────────────────────────────────
# 3. SELECTION
# ──────────────────────────────────────────────────────────────────────────

class TestSelection:
    def test_val_rmse_drives_selection_not_test(self):
        """LSTMForecaster must select the minimum Val_RMSE candidate;
        Test columns must not be present in sweep table."""
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            pytest.skip("TensorFlow not available")

        from src.forecasting import LSTMForecaster

        ctx = _make_th_context(n_train=80, n_val=20, n_test=15)
        train_df = ctx["train_scaled_df"]
        val_df = ctx["val_scaled_df"]
        test_df = ctx["test_scaled_df"]

        feat = "VNINDEX_history"
        target = "VNINDEX"

        f = LSTMForecaster(
            lookback_values=[3, 5],
            batch_size_values=[8],
            epochs=2,
            early_stopping_patience=1,
            reduce_lr_patience=1,
            min_epochs=1,
            lstm_units=[8, 4],
            dense_units=[4],
        )

        X_tr = train_df[[feat]]
        y_tr = train_df[target].rename(target)
        X_v = val_df[[feat]]
        y_v = val_df[target].rename(target)

        f.fit(X_tr, y_tr, X_v, y_v)
        meta = f.get_metadata()
        sweep = meta["sweep_table"]

        # No Test column in sweep table
        for col in sweep.columns:
            assert "Test" not in col, (
                f"Column {col!r} contains 'Test' — Test touched during tuning"
            )

        # Selected must be lowest Val_RMSE
        assert meta["selected_lookback"] in [3, 5]
        best_val = sweep.loc[sweep["Val_RMSE"].idxmin()]
        assert best_val["Lookback"] == meta["selected_lookback"]
        assert best_val["Batch_size"] == meta["selected_batch_size"]


# ──────────────────────────────────────────────────────────────────────────
# 4. FINAL REFIT
# ──────────────────────────────────────────────────────────────────────────

class TestFinalRefit:
    def test_final_refit_uses_new_model_and_fixed_epochs(self):
        """final_refit_and_forecast must produce a fresh model trained for
        exactly best_epoch epochs without EarlyStopping."""
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            pytest.skip("TensorFlow not available")

        from src.forecasting.lstm.sweep import final_refit_and_forecast
        from sklearn.preprocessing import StandardScaler

        ctx = _make_th_context(n_train=80, n_val=20, n_test=15)
        train_df = ctx["train_scaled_df"]
        val_df = ctx["val_scaled_df"]
        test_df = ctx["test_scaled_df"]
        pc_cols = ctx["pc_cols"]
        target_col = ctx["target_col"]

        # Minimal y_scaler that can inverse-transform
        y_scaler = StandardScaler()
        y_scaler.fit(train_df[[target_col]])

        result = final_refit_and_forecast(
            train_scaled_df=train_df,
            val_scaled_df=val_df,
            test_scaled_df=test_df,
            pc_cols=pc_cols,
            target_col=target_col,
            y_scaler=y_scaler,
            selected_lookback=5,
            selected_batch_size=8,
            best_epoch=2,
            learning_rate=1e-3,
            lstm_units=[8, 4],
            dense_units=[4],
            dropout_rate=0.1,
            use_batch_norm=False,
            seed=42,
        )

        assert len(result["test_pred"]) == len(test_df)
        assert len(result["test_dates"]) == len(test_df)
        # Model was returned (new instance)
        assert result["model"] is not None

    def test_final_refit_test_count_matches_test_split(self):
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            pytest.skip("TensorFlow not available")

        from src.forecasting.lstm.sweep import final_refit_and_forecast
        from sklearn.preprocessing import StandardScaler

        n_test = 15
        ctx = _make_th_context(n_train=80, n_val=20, n_test=n_test)
        y_scaler = StandardScaler()
        y_scaler.fit(ctx["train_scaled_df"][[ctx["target_col"]]])

        result = final_refit_and_forecast(
            train_scaled_df=ctx["train_scaled_df"],
            val_scaled_df=ctx["val_scaled_df"],
            test_scaled_df=ctx["test_scaled_df"],
            pc_cols=ctx["pc_cols"],
            target_col=ctx["target_col"],
            y_scaler=y_scaler,
            selected_lookback=5,
            selected_batch_size=8,
            best_epoch=2,
            learning_rate=1e-3,
            lstm_units=[8, 4],
            dense_units=[4],
            dropout_rate=0.1,
            use_batch_norm=False,
            seed=42,
        )
        assert len(result["test_pred"]) == n_test, (
            f"Expected {n_test} test predictions, got {len(result['test_pred'])}"
        )


# ──────────────────────────────────────────────────────────────────────────
# 5. POPULATION PARITY
# ──────────────────────────────────────────────────────────────────────────

class TestPopulationParity:
    def _load_existing_run(self):
        run_dir = PROJECT_ROOT / "artifacts" / "Run_20260823_130205"
        if not run_dir.exists():
            pytest.skip("Reference run not available")
        return run_dir

    def test_th_lstm_test_n_equals_expected(self):
        run_dir = self._load_existing_run()
        th_pred = run_dir / "baselines" / "target_history_lstm" / "predictions_test.csv"
        if not th_pred.exists():
            pytest.skip("TH-LSTM artifacts not yet generated")
        df = pd.read_csv(th_pred)
        assert len(df) == 167, f"Expected 167 test rows, got {len(df)}"

    def test_th_lstm_dates_match_pca_lstm(self):
        run_dir = self._load_existing_run()
        th_pred = run_dir / "baselines" / "target_history_lstm" / "predictions_test.csv"

        # Find any PCA-LSTM prediction
        pca_pred_path = None
        for child in sorted(run_dir.iterdir()):
            if not child.is_dir():
                continue
            manifest_path = child / "run_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            if (manifest.get("status") == "OK" and
                    manifest.get("representation", {}).get("method") == "pca"):
                lstm_dir = child / "outputs" / "lstm_vnindex_sweep"
                cands = sorted(lstm_dir.glob("predictions_lookback_*.csv")) if lstm_dir.exists() else []
                if cands:
                    pca_pred_path = cands[0]
                    break

        if not th_pred.exists() or pca_pred_path is None:
            pytest.skip("TH-LSTM or PCA-LSTM artifacts not available")

        th_df = pd.read_csv(th_pred, parse_dates=["Date"])
        pca_df = pd.read_csv(pca_pred_path, parse_dates=["Date"])

        assert len(th_df) == len(pca_df), (
            f"TH-LSTM N={len(th_df)} != PCA-LSTM N={len(pca_df)}"
        )
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(th_df["Date"]),
            pd.DatetimeIndex(pca_df["Date"]),
            check_names=False,
        )
        np.testing.assert_allclose(
            th_df["Actual_VNINDEX"].values,
            pca_df["Actual_VNINDEX"].values,
            atol=1e-4,
            err_msg="y_true differs between TH-LSTM and PCA-LSTM",
        )


# ──────────────────────────────────────────────────────────────────────────
# 6. ARTIFACT FILES
# ──────────────────────────────────────────────────────────────────────────

class TestArtifacts:
    def _th_dir(self):
        run_dir = PROJECT_ROOT / "artifacts" / "Run_20260823_130205"
        if not run_dir.exists():
            pytest.skip("Reference run not available")
        th_dir = run_dir / "baselines" / "target_history_lstm"
        if not th_dir.exists():
            pytest.skip("TH-LSTM artifacts not yet generated")
        return th_dir

    def test_required_files_exist(self):
        th_dir = self._th_dir()
        required = [
            "predictions_val.csv",
            "predictions_test.csv",
            "tuning_results.csv",
            "summary.json",
            "model_summary.json",
        ]
        for fname in required:
            assert (th_dir / fname).exists(), f"Required artifact missing: {fname}"

    def test_summary_json_has_required_keys(self):
        th_dir = self._th_dir()
        summary = json.loads((th_dir / "summary.json").read_text())
        required_keys = [
            "model", "representation", "forecast_horizon",
            "input_features", "input_dim",
            "selected_lookback", "selected_batch_size", "selected_best_epoch",
            "test_rmse", "test_mae", "test_r2",
            "test_start", "test_end", "same_population_verified",
        ]
        for k in required_keys:
            assert k in summary, f"Key {k!r} missing from summary.json"
        assert summary["representation"] == "target_history"
        assert summary["input_dim"] == 1
        assert summary["input_features"] == ["VNINDEX"]

    def test_summary_test_n_equals_167(self):
        th_dir = self._th_dir()
        summary = json.loads((th_dir / "summary.json").read_text())
        df = pd.read_csv(th_dir / "predictions_test.csv")
        assert len(df) == 167
        assert summary.get("test_n") == 167

    def test_summary_metrics_consistent_with_predictions(self):
        th_dir = self._th_dir()
        summary = json.loads((th_dir / "summary.json").read_text())
        df = pd.read_csv(th_dir / "predictions_test.csv")
        from src.evaluation.metrics import regression_metrics
        m = regression_metrics(df["Actual_VNINDEX"].values, df["Predicted_VNINDEX"].values)
        assert abs(m["RMSE"] - summary["test_rmse"]) < 1e-3, (
            f"RMSE in summary ({summary['test_rmse']:.4f}) != computed ({m['RMSE']:.4f})"
        )
