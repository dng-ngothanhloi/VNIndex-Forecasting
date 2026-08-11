"""
compare_representations.py – Representation comparison table reader
====================================================================
Reads completed run manifests and output artifacts from artifacts/Run_*/
to produce a summary table comparing NoReduction vs PCA at various CEV
thresholds. Does NOT run any models — only aggregates existing results.

Usage:
    python -m src.evaluation.compare_representations
    python -m src.evaluation.compare_representations --artifacts-dir artifacts/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_manifest(run_dir: Path) -> dict | None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_pca_metrics(run_dir: Path) -> dict:
    """Load pca_metrics.csv from run artifacts."""
    pca_metrics_path = run_dir / "data" / "processed" / "pca" / "pca_metrics.csv"
    if not pca_metrics_path.exists():
        return {}
    try:
        df = pd.read_csv(pca_metrics_path, index_col=0)
        return df["value"].to_dict()
    except Exception:
        return {}


def _load_ardl_test_rmse(run_dir: Path) -> dict:
    """Load ARDL test metrics from sweep/forecast artifacts."""
    # Try chapter4 forecast file
    forecast_dir = run_dir / "outputs" / "ardl_vnindex_forecast"
    if not forecast_dir.exists():
        return {}

    # Try reading from the forecast CSV to compute metrics
    for csv_name in ["chapter4_ardl_forecast.csv"]:
        csv_path = forecast_dir / csv_name
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "Actual_VNINDEX" in df.columns and "Predicted_VNINDEX" in df.columns:
                    import numpy as np
                    from sklearn.metrics import mean_absolute_error, r2_score
                    actual = df["Actual_VNINDEX"].values
                    pred = df["Predicted_VNINDEX"].values
                    residuals = actual - pred
                    rmse = float(np.sqrt(np.mean(residuals ** 2)))
                    mae = float(mean_absolute_error(actual, pred))
                    mape = float(np.mean(np.abs(residuals / (actual + 1e-8))) * 100)
                    r2 = float(r2_score(actual, pred))
                    return {"ARDL_RMSE": rmse, "ARDL_MAE": mae, "ARDL_MAPE": mape, "ARDL_R2": r2,
                            "ARDL_N": len(df)}
            except Exception:
                pass

    # Try sweep results for selected pair info
    sweep_path = run_dir / "outputs" / "ardl_vnindex_pca_sweep" / "sweep_results.csv"
    info = {}
    if sweep_path.exists():
        try:
            sweep = pd.read_csv(sweep_path)
            ok = sweep[sweep["Status"] == "OK"]
            if not ok.empty:
                best = ok.loc[ok["BIC"].idxmin()]
                info["ARDL_P"] = int(best["P"])
                info["ARDL_Q"] = int(best["Q"])
        except Exception:
            pass
    return info


def _load_lstm_test_rmse(run_dir: Path) -> dict:
    """Load LSTM test metrics from sweep artifacts."""
    lstm_dir = run_dir / "outputs" / "lstm_vnindex_sweep"
    if not lstm_dir.exists():
        return {}

    # Find best model from sweep_summary
    summary_path = lstm_dir / "sweep_summary.csv"
    info = {}
    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path)
            best_row = summary.loc[summary["Val_RMSE"].idxmin()]
            info["LSTM_LB"] = int(best_row["Lookback"])
            info["LSTM_BS"] = int(best_row["Batch_size"])
            info["LSTM_best_epoch"] = int(best_row["Best_Epoch"])
        except Exception:
            pass

    # Find predictions file for test metrics
    for pred_file in sorted(lstm_dir.glob("predictions_lookback_*.csv")):
        try:
            df = pd.read_csv(pred_file)
            if "Actual_VNINDEX" in df.columns and "Predicted_VNINDEX" in df.columns:
                import numpy as np
                from sklearn.metrics import mean_absolute_error, r2_score
                actual = df["Actual_VNINDEX"].values
                pred = df["Predicted_VNINDEX"].values
                residuals = actual - pred
                rmse = float(np.sqrt(np.mean(residuals ** 2)))
                mae = float(mean_absolute_error(actual, pred))
                mape = float(np.mean(np.abs(residuals / (actual + 1e-8))) * 100)
                r2 = float(r2_score(actual, pred))
                info.update({"LSTM_RMSE": rmse, "LSTM_MAE": mae, "LSTM_MAPE": mape,
                             "LSTM_R2": r2, "LSTM_N": len(df)})
                break
        except Exception:
            continue

    return info


def _load_dm_results(run_dir: Path) -> dict:
    """Load DM test p-values."""
    dm_path = run_dir / "outputs" / "model_comparison" / "dm_test_results.csv"
    if not dm_path.exists():
        return {}
    try:
        dm = pd.read_csv(dm_path)
        info = {}
        for _, row in dm.iterrows():
            loss = row.get("Loss_Type", "")
            if loss == "MSE":
                info["DM_MSE_p"] = row.get("p_value")
            elif loss == "MAE":
                info["DM_MAE_p"] = row.get("p_value")
        return info
    except Exception:
        return {}


def collect_run_results(artifacts_dir: Path) -> pd.DataFrame:
    """Scan all Run_* directories and aggregate results into a table."""
    rows = []

    run_dirs = sorted(artifacts_dir.glob("Run_*"))
    if not run_dirs:
        print(f"[WARN] No Run_* directories found in {artifacts_dir}")
        return pd.DataFrame()

    for run_dir in run_dirs:
        manifest = _load_manifest(run_dir)
        if manifest is None:
            continue
        if manifest.get("status") != "OK":
            continue

        # Determine representation
        repr_info = manifest.get("representation", {})
        method = repr_info.get("method", manifest.get("pca", {}).get("threshold", "pca"))
        if isinstance(method, float):
            method = "pca"

        cev_requested = repr_info.get("cev_requested",
                                       manifest.get("pca", {}).get("threshold"))

        pca_metrics = _load_pca_metrics(run_dir)
        ardl_info = _load_ardl_test_rmse(run_dir)
        lstm_info = _load_lstm_test_rmse(run_dir)
        dm_info = _load_dm_results(run_dir)

        row = {
            "run": run_dir.name,
            "representation": method if method else "pca",
            "cev_requested": cev_requested,
            "k": pca_metrics.get("k_optimal"),
            "dim_reduction_pct": pca_metrics.get("dim_reduction_pct"),
        }
        row.update(ardl_info)
        row.update(lstm_info)
        row.update(dm_info)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort: none first, then by cev_requested ascending
    df["_sort"] = df["cev_requested"].fillna(-1).astype(float)
    df = df.sort_values(["representation", "_sort"]).drop(columns=["_sort"])

    return df


def print_comparison_table(df: pd.DataFrame) -> None:
    """Print a formatted comparison table."""
    if df.empty:
        print("[INFO] No completed runs found to compare.")
        return

    # Select display columns
    display_cols = [
        "representation", "cev_requested", "k", "dim_reduction_pct",
        "ARDL_RMSE", "LSTM_RMSE", "ARDL_MAE", "LSTM_MAE",
        "ARDL_R2", "LSTM_R2", "DM_MSE_p", "DM_MAE_p",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    print("\n" + "=" * 100)
    print("  REPRESENTATION COMPARISON TABLE")
    print("=" * 100)
    print(df[display_cols].to_string(index=False, float_format="%.4f"))
    print("=" * 100)

    # Additional detail columns if available
    detail_cols = ["ARDL_P", "ARDL_Q", "LSTM_LB", "LSTM_BS", "LSTM_best_epoch"]
    detail_cols = [c for c in detail_cols if c in df.columns]
    if detail_cols:
        print("\n  MODEL SELECTION DETAIL:")
        print(df[["representation", "cev_requested"] + detail_cols].to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare representation results across runs")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts",
                        help="Path to artifacts directory containing Run_* subdirs")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Optional: save table to CSV")
    args = parser.parse_args()

    artifacts_dir = PROJECT_ROOT / args.artifacts_dir
    if not artifacts_dir.exists():
        print(f"[FAIL] Artifacts directory not found: {artifacts_dir}")
        sys.exit(1)

    df = collect_run_results(artifacts_dir)
    print_comparison_table(df)

    if args.output_csv:
        out_path = PROJECT_ROOT / args.output_csv
        df.to_csv(out_path, index=False)
        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
