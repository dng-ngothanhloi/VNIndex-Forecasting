#!/usr/bin/env python
"""
run_target_history_lstm_baseline.py – Target-History LSTM baseline
====================================================================
Univariate LSTM: [VNINDEX_{t-L}, ..., VNINDEX_{t-1}] → VNINDEX_t.
No PCA components, no stock features.

Reuses the canonical LSTM sweep/refit/windowing pipeline unchanged.
Only the input representation is changed: a single VNINDEX history
channel replaces PCA columns.

Usage:
    python experiments/run_target_history_lstm_baseline.py \\
        --run-dir artifacts/Run_20260823_130205

Artifacts written to:
    <run_dir>/baselines/target_history_lstm/
        predictions_val.csv
        predictions_test.csv
        tuning_results.csv
        selected_tuning_history.csv
        summary.json
        model_summary.json
        multiseed_results.csv
        multiseed_summary.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.lstm.setup import run_imports, run_paths
from src.forecasting.lstm.data import run_load_data, run_prepare_data
from src.forecasting.lstm.sweep import (
    prepare_target_history_context,
    run_train_and_evaluate,
    final_refit_and_forecast,
)
from src.evaluation.metrics import regression_metrics


GOVERNED_SEEDS = [42, 52, 62, 72, 82]
REFERENCE_SEED = 42
ARTIFACT_SUBDIR = "baselines/target_history_lstm"
EXPECTED_TEST_N = 167
EXPECTED_VAL_N = 123


def _hdr(text: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {text}")
    print("=" * 72)


def _load_config() -> dict:
    import yaml
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_pca_context_for_run(run_dir: Path) -> dict:
    """Bootstrap an LSTM context from a specific Run_* child's processed data.

    Reads PCA data from the first successful pca_cev_* child so the
    split/scaler/dates are identical to the PCA-LSTM experiment.  The
    target-history transform is then applied on top, giving the correct
    chronological boundaries.
    """
    # Find first successful pca child to borrow processed data paths from
    child_dir: Path | None = None
    for cd in sorted(run_dir.iterdir()):
        if not cd.is_dir():
            continue
        manifest_path = cd / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "OK":
            continue
        repr_info = manifest.get("representation", {})
        if repr_info.get("method") == "pca":
            child_dir = cd
            break

    if child_dir is None:
        raise FileNotFoundError(
            f"No successful PCA child found in {run_dir}. "
            "Run the full sweep first: python experiments/run_experiment.py --full-sweep"
        )

    pca_dir = child_dir / "data" / "processed" / "pca"
    core_dir = child_dir / "data" / "processed" / "core"

    if not pca_dir.exists() or not core_dir.exists():
        raise FileNotFoundError(
            f"Expected data dirs not found under {child_dir}. "
            "Ensure the sweep was run with --include-multiseed or the child "
            "snapshot was generated."
        )

    print(f"  Using processed data from child: {child_dir.name}")
    context: dict = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "PCA_DIR": pca_dir,
        "CORE_DIR": core_dir,
    }
    context = run_imports(context)
    context = run_load_data(context)
    context = run_prepare_data(context)
    return context


def run_reference_tuning(run_dir: Path) -> dict:
    """Run the tuning sweep once (seed=42) on target-history representation."""
    _hdr(f"TARGET-HISTORY LSTM — REFERENCE TUNING (seed={REFERENCE_SEED})")
    pca_context = _make_pca_context_for_run(run_dir)
    th_context = prepare_target_history_context(pca_context)

    # Redirect sweep output to our dedicated directory
    out_dir = run_dir / ARTIFACT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    th_context["PROJECT_ROOT"] = PROJECT_ROOT

    th_context = run_train_and_evaluate(th_context)
    return th_context


def run_multiseed(context: dict, out_dir: Path) -> pd.DataFrame:
    """Refit on frozen hyperparameters for all 5 seeds."""
    cfg = _load_config()
    lstm_cfg = cfg.get("lstm", {})

    selected_lookback = context["selected_lookback"]
    selected_batch_size = context["selected_batch_size"]
    best_epoch = context["final_refit_epochs"]

    _hdr(f"MULTI-SEED STABILITY  LB={selected_lookback} BS={selected_batch_size} "
         f"epochs={best_epoch}")

    rows = []
    test_dates_ref = None

    import tensorflow as tf

    for seed in GOVERNED_SEEDS:
        tf.keras.backend.clear_session()
        print(f"\n[SEED {seed}] refitting NEW model …")
        result = final_refit_and_forecast(
            train_scaled_df=context["train_scaled_df"],
            val_scaled_df=context["val_scaled_df"],
            test_scaled_df=context["test_scaled_df"],
            pc_cols=context["pc_cols"],
            target_col=context["target_col"],
            y_scaler=context["y_scaler"],
            selected_lookback=selected_lookback,
            selected_batch_size=selected_batch_size,
            best_epoch=best_epoch,
            learning_rate=lstm_cfg.get("learning_rate", 3e-4),
            lstm_units=lstm_cfg.get("lstm_units", [32, 16]),
            dense_units=lstm_cfg.get("dense_units", [8]),
            dropout_rate=lstm_cfg.get("dropout_rate", 0.1),
            use_batch_norm=lstm_cfg.get("use_batch_norm", False),
            seed=seed,
        )

        test_dates = result["test_dates"]
        if test_dates_ref is None:
            test_dates_ref = test_dates
        else:
            assert test_dates.equals(test_dates_ref), (
                f"[FAIL-FAST] seed={seed} Test dates differ from seed={REFERENCE_SEED}. "
                "All seeds must forecast the same population."
            )

        m = regression_metrics(result["y_test_true"], result["test_pred"])
        rows.append({"Seed": seed, "N_Test": len(test_dates), **m})
        print(f"[SEED {seed}] RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R2={m['R2']:.4f}")

        pd.DataFrame({
            "Date": test_dates,
            "Actual_VNINDEX": result["y_test_true"],
            "Predicted_VNINDEX": result["test_pred"],
            "Residual": result["y_test_true"] - result["test_pred"],
        }).to_csv(out_dir / f"predictions_seed_{seed}.csv", index=False)

        del result
        tf.keras.backend.clear_session()

    summary = pd.DataFrame(rows)
    return summary


def _compute_multiseed_agg(summary: pd.DataFrame) -> dict:
    agg = {}
    for metric in ("RMSE", "MAE", "MAPE(%)", "R2"):
        if metric not in summary.columns:
            continue
        agg[metric] = {
            "mean": float(summary[metric].mean()),
            "std": float(summary[metric].std(ddof=1)),
            "min": float(summary[metric].min()),
            "max": float(summary[metric].max()),
        }
    return agg


def save_artifacts(context: dict, out_dir: Path) -> None:
    """Save all required artifacts from reference tuning."""
    # Retrieve predictions from the reference (seed=42) final refit
    sweep_dir = PROJECT_ROOT / "outputs" / "lstm_vnindex_sweep"
    selected_lookback = context["selected_lookback"]
    selected_batch_size = context["selected_batch_size"]
    pred_src = sweep_dir / f"predictions_lookback_{selected_lookback}_batch_{selected_batch_size}.csv"

    if pred_src.exists():
        shutil.copy2(pred_src, out_dir / "predictions_test.csv")
        pred_test_df = pd.read_csv(out_dir / "predictions_test.csv")
    else:
        raise FileNotFoundError(f"Reference test predictions not found: {pred_src}")

    # Val predictions: get from selected val dates in context
    val_scaled_df = context["val_scaled_df"]
    target_col = context["target_col"]
    y_scaler = context["y_scaler"]
    pc_cols = context["pc_cols"]

    from src.forecasting.lstm.data import make_cross_boundary_windowed_data, add_target_history
    import numpy as np
    train_scaled_df = context["train_scaled_df"]

    X_val, y_val, X_val_hist, val_dates = make_cross_boundary_windowed_data(
        train_scaled_df, val_scaled_df, pc_cols, target_col, selected_lookback,
    )
    X_val_final = np.concatenate([X_val, X_val_hist], axis=2)

    selected_model = context["selected_model"]
    val_pred_scaled = selected_model.predict(X_val_final, verbose=0)
    val_pred = y_scaler.inverse_transform(val_pred_scaled).ravel()
    y_val_true = y_scaler.inverse_transform(y_val).ravel()

    pd.DataFrame({
        "Date": val_dates,
        "Actual_VNINDEX": y_val_true,
        "Predicted_VNINDEX": val_pred,
        "Residual": y_val_true - val_pred,
    }).to_csv(out_dir / "predictions_val.csv", index=False)

    val_m = regression_metrics(y_val_true, val_pred)

    # tuning_results
    sweep_csv = sweep_dir / "sweep_summary.csv"
    if sweep_csv.exists():
        shutil.copy2(sweep_csv, out_dir / "tuning_results.csv")

    # selected_tuning_history
    sel_hist = sweep_dir / "selected_tuning_history.csv"
    if sel_hist.exists():
        shutil.copy2(sel_hist, out_dir / "selected_tuning_history.csv")
    else:
        # fallback: copy from tuning_history/
        hist_src = (sweep_dir / "tuning_history" /
                    f"tuning_history_LB{selected_lookback}_BS{selected_batch_size}.csv")
        if hist_src.exists():
            shutil.copy2(hist_src, out_dir / "selected_tuning_history.csv")

    # Compute test metrics for seed=42
    test_m = regression_metrics(
        pred_test_df["Actual_VNINDEX"].values,
        pred_test_df["Predicted_VNINDEX"].values,
    )

    cfg = _load_config()
    lstm_cfg = cfg.get("lstm", {})

    selected_metrics_row = context.get("selected_metrics_row", {}) or {}

    summary = {
        "model": "target_history_lstm",
        "representation": "target_history",
        "forecast_horizon": "T+1",
        "input_features": ["VNINDEX"],
        "input_dim": 1,
        "selected_lookback": selected_lookback,
        "selected_batch_size": selected_batch_size,
        "selected_best_epoch": context.get("final_refit_epochs"),
        "architecture": {
            "lstm_units": lstm_cfg.get("lstm_units", [32, 16]),
            "dense_units": lstm_cfg.get("dense_units", [8]),
            "dropout_rate": lstm_cfg.get("dropout_rate", 0.1),
            "use_batch_norm": lstm_cfg.get("use_batch_norm", False),
        },
        "learning_rate": lstm_cfg.get("learning_rate", 3e-4),
        "dropout": lstm_cfg.get("dropout_rate", 0.1),
        "optimizer": "Adam",
        "loss": "mse",
        "train_n": len(train_scaled_df),
        "val_n": len(val_scaled_df),
        "test_n": len(pred_test_df),
        "val_rmse": val_m["RMSE"],
        "val_mae": val_m["MAE"],
        "test_rmse": test_m["RMSE"],
        "test_mae": test_m["MAE"],
        "test_mape": test_m["MAPE(%)"],
        "test_r2": test_m["R2"],
        "test_start": str(pred_test_df["Date"].min())[:10],
        "test_end": str(pred_test_df["Date"].max())[:10],
        "same_population_verified": len(pred_test_df) == EXPECTED_TEST_N,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    model_summary = {
        "model": "target_history_lstm",
        "input_channels": ["VNINDEX_history"],
        "input_dim_per_timestep": 1,
        "lookback": selected_lookback,
        "batch_size": selected_batch_size,
        "best_epoch_tuning": selected_metrics_row.get("Best_Epoch"),
        "final_refit_epochs": context.get("final_refit_epochs"),
        "architecture": summary["architecture"],
        "learning_rate": summary["learning_rate"],
    }
    with open(out_dir / "model_summary.json", "w") as f:
        json.dump(model_summary, f, indent=2)

    return test_m, val_m


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-History LSTM baseline")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--skip-multiseed", action="store_true",
                        help="Skip the 5-seed stability evaluation")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        print(f"[FAIL] Run dir not found: {run_dir}")
        sys.exit(1)

    out_dir = run_dir / ARTIFACT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reference tuning (seed=42)
    context = run_reference_tuning(run_dir)

    # 2. Population invariant checks
    selected_lookback = context["selected_lookback"]
    test_pred_path = (
        PROJECT_ROOT / "outputs" / "lstm_vnindex_sweep"
        / f"predictions_lookback_{selected_lookback}_batch_{context['selected_batch_size']}.csv"
    )
    if test_pred_path.exists():
        check_df = pd.read_csv(test_pred_path)
        n_test = len(check_df)
        assert n_test == EXPECTED_TEST_N, (
            f"[FAIL] Test N={n_test} != expected {EXPECTED_TEST_N}. "
            "Population invariant violated."
        )
        print(f"[OK] Test population: N={n_test} ✓")

    # 3. Save artifacts
    _hdr("SAVING ARTIFACTS")
    test_m, val_m = save_artifacts(context, out_dir)
    print(f"  Val  RMSE={val_m['RMSE']:.4f}  MAE={val_m['MAE']:.4f}")
    print(f"  Test RMSE={test_m['RMSE']:.4f}  MAE={test_m['MAE']:.4f}  R2={test_m['R2']:.4f}")

    # 4. Multi-seed stability
    if not args.skip_multiseed:
        multiseed_summary = run_multiseed(context, out_dir)
        multiseed_summary.to_csv(out_dir / "multiseed_results.csv", index=False)
        agg = _compute_multiseed_agg(multiseed_summary)
        with open(out_dir / "multiseed_summary.json", "w") as f:
            json.dump(agg, f, indent=2)
        _hdr("MULTI-SEED SUMMARY")
        for metric, stats in agg.items():
            print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}  "
                  f"[{stats['min']:.4f}, {stats['max']:.4f}]")
    else:
        print("[SKIP] Multi-seed evaluation skipped (--skip-multiseed)")

    _hdr("DONE")
    print(f"  Artifacts: {out_dir}")
    print(f"  Test RMSE (seed=42): {test_m['RMSE']:.4f}")


if __name__ == "__main__":
    main()
