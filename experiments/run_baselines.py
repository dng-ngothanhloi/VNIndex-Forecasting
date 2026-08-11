"""
run_baselines.py – Persistence + AR(1) baselines against existing PCA-ARDL
============================================================================
Reads VNINDEX from an existing sweep Run_*, computes persistence and AR(1)
baselines, and produces comparison tables + DM tests.

Usage:
    python experiments/run_baselines.py --run-dir artifacts/Run_20260811_134344
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.ar import persistence_forecast, fit_ar1, ar1_rolling_forecast
from src.evaluation.dm_test import diebold_mariano_test
from src.evaluation.metrics import regression_metrics


def _hdr(text: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {text}")
    print("=" * 72)


def _load_vnindex(run_dir: Path) -> pd.Series:
    """Load VNINDEX from the first available child's core data."""
    for child in sorted(run_dir.iterdir()):
        core_path = child / "data" / "processed" / "core" / "vnindex_target.csv"
        if core_path.exists():
            df = pd.read_csv(core_path, parse_dates=["Ngày"]).set_index("Ngày")
            return df["VNINDEX"]
    raise FileNotFoundError(f"No vnindex_target.csv found in any child of {run_dir}")


def _load_split_dates(run_dir: Path) -> dict:
    """Load split boundaries from first available child."""
    for child in sorted(run_dir.iterdir()):
        split_path = child / "data" / "processed" / "splits" / "split_summary.csv"
        if split_path.exists():
            df = pd.read_csv(split_path)
            result = {}
            for _, row in df.iterrows():
                result[row["Split"].strip()] = {
                    "from": pd.Timestamp(row["From"]),
                    "to": pd.Timestamp(row["To"]),
                    "rows": int(row["Rows"]),
                }
            return result
    raise FileNotFoundError(f"No split_summary.csv found in any child of {run_dir}")


def _load_ardl_forecasts(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Load existing ARDL forecasts from each pca_cev_* child."""
    results = {}
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        manifest_path = child / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "OK":
            continue
        repr_info = manifest.get("representation", {})
        if repr_info.get("method") != "pca":
            continue

        forecast_path = child / "outputs" / "ardl_vnindex_forecast" / "chapter4_ardl_forecast.csv"
        if forecast_path.exists():
            df = pd.read_csv(forecast_path, parse_dates=["Date"])
            label = child.name
            results[label] = {
                "df": df,
                "cev": repr_info.get("cev_requested"),
                "k": repr_info.get("k_actual"),
                "manifest": manifest,
            }
    return results


def main():
    parser = argparse.ArgumentParser(description="Persistence + AR(1) baselines")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to a specific artifacts/Run_* directory (REQUIRED)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        print(f"[FAIL] Run directory not found: {run_dir}")
        sys.exit(1)

    _hdr("PERSISTENCE + AR(1) BASELINES")
    print(f"  Run dir: {run_dir}")

    # ── Load data ─────────────────────────────────────────────────
    vnindex = _load_vnindex(run_dir)
    splits = _load_split_dates(run_dir)
    print(f"  VNINDEX: {len(vnindex)} observations ({vnindex.index.min().date()} → {vnindex.index.max().date()})")
    print(f"  Train: {splits['Train']['rows']} | Val: {splits['Validation']['rows']} | Test: {splits['Test']['rows']}")

    # Split VNINDEX
    train_end = splits["Train"]["to"]
    val_start = splits["Validation"]["from"]
    val_end = splits["Validation"]["to"]
    test_start = splits["Test"]["from"]

    y_train = vnindex[vnindex.index <= train_end]
    y_val = vnindex[(vnindex.index >= val_start) & (vnindex.index <= val_end)]
    y_test = vnindex[vnindex.index >= test_start]
    y_trainval = vnindex[vnindex.index <= val_end]

    assert len(y_train) == splits["Train"]["rows"], f"Train mismatch: {len(y_train)} vs {splits['Train']['rows']}"
    assert len(y_val) == splits["Validation"]["rows"]
    assert len(y_test) == splits["Test"]["rows"]

    # ── Persistence ───────────────────────────────────────────────
    _hdr("PERSISTENCE BASELINE")
    persist_val = persistence_forecast(vnindex, y_val.index)
    persist_test = persistence_forecast(vnindex, y_test.index)

    persist_val_metrics = regression_metrics(y_val.values, persist_val.values)
    persist_test_metrics = regression_metrics(y_test.values, persist_test.values)

    print(f"  Val:  RMSE={persist_val_metrics['RMSE']:.4f} MAE={persist_val_metrics['MAE']:.4f}")
    print(f"  Test: RMSE={persist_test_metrics['RMSE']:.4f} MAE={persist_test_metrics['MAE']:.4f} "
          f"R2={persist_test_metrics['R2']:.4f}")

    # ── AR(1) ─────────────────────────────────────────────────────
    _hdr("AR(1) BASELINE")

    # Val diagnostic: fit on Train only
    ar1_train_coef = fit_ar1(y_train)
    ar1_val_pred = ar1_rolling_forecast(ar1_train_coef, vnindex, y_val.index)
    ar1_val_metrics = regression_metrics(y_val.values, ar1_val_pred.values)
    print(f"  Train-fit: const={ar1_train_coef['const']:.4f} phi={ar1_train_coef['phi']:.6f} "
          f"(n={ar1_train_coef['n_obs']})")
    print(f"  Val:  RMSE={ar1_val_metrics['RMSE']:.4f} MAE={ar1_val_metrics['MAE']:.4f}")

    # Final: refit on Train+Val
    ar1_dev_coef = fit_ar1(y_trainval)
    ar1_test_pred = ar1_rolling_forecast(ar1_dev_coef, vnindex, y_test.index)
    ar1_test_metrics = regression_metrics(y_test.values, ar1_test_pred.values)
    print(f"  Dev-fit:   const={ar1_dev_coef['const']:.4f} phi={ar1_dev_coef['phi']:.6f} "
          f"(n={ar1_dev_coef['n_obs']})")
    print(f"  Test: RMSE={ar1_test_metrics['RMSE']:.4f} MAE={ar1_test_metrics['MAE']:.4f} "
          f"R2={ar1_test_metrics['R2']:.4f}")

    # ── Save baseline artifacts ───────────────────────────────────
    baselines_dir = run_dir / "baselines"

    # Persistence
    persist_dir = baselines_dir / "persistence"
    persist_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Date": y_val.index, "Actual_VNINDEX": y_val.values,
                   "Predicted_VNINDEX": persist_val.values}).to_csv(
        persist_dir / "predictions_val.csv", index=False)
    pd.DataFrame({"Date": y_test.index, "Actual_VNINDEX": y_test.values,
                   "Predicted_VNINDEX": persist_test.values}).to_csv(
        persist_dir / "predictions_test.csv", index=False)
    persist_summary = {
        "model": "persistence", "forecast_horizon": "T+1", "fit_required": False,
        "val_n": len(y_val), "test_n": len(y_test),
        "val_rmse": persist_val_metrics["RMSE"], "val_mae": persist_val_metrics["MAE"],
        "test_rmse": persist_test_metrics["RMSE"], "test_mae": persist_test_metrics["MAE"],
        "test_mape": persist_test_metrics["MAPE(%)"], "test_r2": persist_test_metrics["R2"],
        "test_start": str(y_test.index.min().date()),
        "test_end": str(y_test.index.max().date()),
        "same_population_verified": True,
    }
    with open(persist_dir / "summary.json", "w") as f:
        json.dump(persist_summary, f, indent=2)

    # AR(1)
    ar1_dir = baselines_dir / "ar1"
    ar1_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Date": y_val.index, "Actual_VNINDEX": y_val.values,
                   "Predicted_VNINDEX": ar1_val_pred.values}).to_csv(
        ar1_dir / "predictions_val.csv", index=False)
    pd.DataFrame({"Date": y_test.index, "Actual_VNINDEX": y_test.values,
                   "Predicted_VNINDEX": ar1_test_pred.values}).to_csv(
        ar1_dir / "predictions_test.csv", index=False)
    ar1_summary = {
        "model": "AR(1)", "forecast_horizon": "T+1",
        "coefficient_const": ar1_dev_coef["const"],
        "coefficient_phi": ar1_dev_coef["phi"],
        "train_fit_n": ar1_train_coef["n_obs"],
        "val_n": len(y_val), "val_rmse": ar1_val_metrics["RMSE"],
        "val_mae": ar1_val_metrics["MAE"],
        "development_fit_n": ar1_dev_coef["n_obs"],
        "test_n": len(y_test),
        "test_rmse": ar1_test_metrics["RMSE"], "test_mae": ar1_test_metrics["MAE"],
        "test_mape": ar1_test_metrics["MAPE(%)"], "test_r2": ar1_test_metrics["R2"],
        "test_start": str(y_test.index.min().date()),
        "test_end": str(y_test.index.max().date()),
        "same_population_verified": True,
    }
    with open(ar1_dir / "summary.json", "w") as f:
        json.dump(ar1_summary, f, indent=2)
    with open(ar1_dir / "model_summary.json", "w") as f:
        json.dump({"train_coef": ar1_train_coef, "dev_coef": ar1_dev_coef}, f, indent=2)

    # ── Load existing ARDL forecasts ──────────────────────────────
    _hdr("LOADING EXISTING PCA-ARDL FORECASTS")
    ardl_results = _load_ardl_forecasts(run_dir)
    print(f"  Found {len(ardl_results)} PCA-ARDL children")

    # Verify same population
    for label, info in ardl_results.items():
        ardl_df = info["df"]
        ardl_dates = pd.DatetimeIndex(ardl_df["Date"])
        assert len(ardl_df) == len(y_test), \
            f"{label}: ARDL N={len(ardl_df)} != Test N={len(y_test)}"
        assert ardl_dates.equals(y_test.index), \
            f"{label}: ARDL dates != Test dates"
        assert np.allclose(ardl_df["Actual_VNINDEX"].values, y_test.values, atol=1e-6), \
            f"{label}: ARDL y_true != baseline y_true"
        print(f"  [{label}] N={len(ardl_df)} dates=OK y_true=OK")

    # ── Baseline comparison table ─────────────────────────────────
    _hdr("BASELINE COMPARISON TABLE")
    comp_rows = [
        {"model": "Persistence", "representation": "none", "cev_requested": None,
         "k": None, **{k: v for k, v in persist_test_metrics.items()}, "N": len(y_test)},
        {"model": "AR(1)", "representation": "none", "cev_requested": None,
         "k": None, **{k: v for k, v in ar1_test_metrics.items()}, "N": len(y_test)},
    ]
    for label, info in sorted(ardl_results.items()):
        ardl_df = info["df"]
        m = regression_metrics(ardl_df["Actual_VNINDEX"].values, ardl_df["Predicted_VNINDEX"].values)
        comp_rows.append({
            "model": f"PCA-ARDL", "representation": "pca",
            "cev_requested": info["cev"], "k": info["k"],
            **{k: v for k, v in m.items()}, "N": len(ardl_df),
        })

    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.to_string(index=False))

    comp_dir = run_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(comp_dir / "baseline_comparison.csv", index=False)

    # ── Incremental value table ───────────────────────────────────
    _hdr("PCA-ARDL INCREMENTAL VALUE")
    incr_rows = []
    for label, info in sorted(ardl_results.items()):
        ardl_df = info["df"]
        ardl_m = regression_metrics(ardl_df["Actual_VNINDEX"].values, ardl_df["Predicted_VNINDEX"].values)
        incr_rows.append({
            "CEV": info["cev"], "k": info["k"],
            "Persistence_RMSE": persist_test_metrics["RMSE"],
            "AR1_RMSE": ar1_test_metrics["RMSE"],
            "PCA_ARDL_RMSE": ardl_m["RMSE"],
            "RMSE_gain_vs_persistence": persist_test_metrics["RMSE"] - ardl_m["RMSE"],
            "RMSE_gain_pct_vs_persistence": (persist_test_metrics["RMSE"] - ardl_m["RMSE"]) / persist_test_metrics["RMSE"] * 100,
            "RMSE_gain_vs_AR1": ar1_test_metrics["RMSE"] - ardl_m["RMSE"],
            "RMSE_gain_pct_vs_AR1": (ar1_test_metrics["RMSE"] - ardl_m["RMSE"]) / ar1_test_metrics["RMSE"] * 100,
            "Persistence_MAE": persist_test_metrics["MAE"],
            "AR1_MAE": ar1_test_metrics["MAE"],
            "PCA_ARDL_MAE": ardl_m["MAE"],
            "MAE_gain_vs_persistence": persist_test_metrics["MAE"] - ardl_m["MAE"],
            "MAE_gain_pct_vs_persistence": (persist_test_metrics["MAE"] - ardl_m["MAE"]) / persist_test_metrics["MAE"] * 100,
            "MAE_gain_vs_AR1": ar1_test_metrics["MAE"] - ardl_m["MAE"],
            "MAE_gain_pct_vs_AR1": (ar1_test_metrics["MAE"] - ardl_m["MAE"]) / ar1_test_metrics["MAE"] * 100,
        })
    incr_df = pd.DataFrame(incr_rows)
    print(incr_df[["CEV", "k", "PCA_ARDL_RMSE", "RMSE_gain_vs_persistence",
                   "RMSE_gain_pct_vs_persistence", "RMSE_gain_vs_AR1",
                   "RMSE_gain_pct_vs_AR1"]].to_string(index=False))
    incr_df.to_csv(comp_dir / "pca_ardl_incremental_value.csv", index=False)

    # ── DM tests ──────────────────────────────────────────────────
    _hdr("DM TESTS: BASELINES vs PCA-ARDL")
    # Sign convention: forecast1=baseline, forecast2=PCA-ARDL
    # negative mean_diff → PCA-ARDL has HIGHER loss (baseline better)
    # positive mean_diff → PCA-ARDL has LOWER loss (PCA-ARDL better)
    # Wait — check actual convention from dm_test.py:
    # d = e1^2 - e2^2; d_bar < 0 → forecast1 better
    # So: forecast1=PCA-ARDL, forecast2=baseline
    # Then: negative d_bar → PCA-ARDL better (lower loss)
    dm_rows = []
    actual = y_test.values

    for label, info in sorted(ardl_results.items()):
        ardl_pred = info["df"]["Predicted_VNINDEX"].values

        for baseline_name, baseline_pred in [("Persistence", persist_test.values),
                                              ("AR(1)", ar1_test_pred.values)]:
            for loss in ("mse", "mae"):
                # forecast1=PCA-ARDL, forecast2=baseline
                # negative dm_stat → PCA-ARDL has lower loss
                dm = diebold_mariano_test(actual, ardl_pred, baseline_pred,
                                          loss_type=loss, alternative="two-sided")
                dm_rows.append({
                    "cev": info["cev"], "k": info["k"],
                    "comparison": f"PCA-ARDL vs {baseline_name}",
                    "loss_type": loss.upper(),
                    "mean_loss_diff": dm["mean_diff"],
                    "dm_stat": dm["dm_stat"],
                    "p_value": dm["p_value"],
                    "significant_5pct": dm["p_value"] < 0.05,
                    "n": dm["sample_size"],
                    "direction": "PCA-ARDL lower loss" if dm["mean_diff"] < 0 else "Baseline lower loss",
                })

    dm_df = pd.DataFrame(dm_rows)
    print(dm_df[["cev", "comparison", "loss_type", "dm_stat", "p_value",
                 "significant_5pct", "direction"]].to_string(index=False))
    dm_df.to_csv(comp_dir / "baseline_dm_comparison.csv", index=False)

    # ── Interpretation ────────────────────────────────────────────
    _hdr("INTERPRETATION")
    interpretations = []
    for label, info in sorted(ardl_results.items()):
        cev = info["cev"]
        ardl_pred = info["df"]["Predicted_VNINDEX"].values
        ardl_m = regression_metrics(actual, ardl_pred)

        for baseline_name, baseline_metrics in [("Persistence", persist_test_metrics),
                                                 ("AR(1)", ar1_test_metrics)]:
            rmse_gain = baseline_metrics["RMSE"] - ardl_m["RMSE"]
            rmse_gain_pct = rmse_gain / baseline_metrics["RMSE"] * 100
            # Find DM result
            dm_row = dm_df[(dm_df["cev"] == cev) &
                           (dm_df["comparison"] == f"PCA-ARDL vs {baseline_name}") &
                           (dm_df["loss_type"] == "MSE")]
            sig = bool(dm_row["significant_5pct"].iloc[0]) if not dm_row.empty else False
            pca_lower = bool(dm_row["mean_loss_diff"].iloc[0] < 0) if not dm_row.empty else False

            if pca_lower and sig and rmse_gain_pct > 10:
                verdict = "material_and_significant"
            elif pca_lower and sig:
                verdict = "small_but_significant"
            elif pca_lower and not sig:
                verdict = "small_not_significant"
            elif not pca_lower and not sig:
                verdict = "no_improvement"
            else:
                verdict = "worse"

            interpretations.append({
                "cev": cev, "vs_baseline": baseline_name,
                "rmse_gain": rmse_gain, "rmse_gain_pct": rmse_gain_pct,
                "dm_significant": sig, "pca_ardl_lower_loss": pca_lower,
                "verdict": verdict,
            })

    interp_df = pd.DataFrame(interpretations)
    print(interp_df[["cev", "vs_baseline", "rmse_gain", "rmse_gain_pct",
                     "dm_significant", "verdict"]].to_string(index=False))

    with open(comp_dir / "baseline_interpretation.json", "w") as f:
        json.dump(interpretations, f, indent=2, default=str)

    # ── Final summary ─────────────────────────────────────────────
    _hdr("DONE")
    print(f"  Artifacts saved to: {baselines_dir}")
    print(f"  Comparison saved to: {comp_dir}")
    print(f"  Persistence Test RMSE: {persist_test_metrics['RMSE']:.4f}")
    print(f"  AR(1) Test RMSE:       {ar1_test_metrics['RMSE']:.4f}")
    for label, info in sorted(ardl_results.items()):
        ardl_m = regression_metrics(actual, info["df"]["Predicted_VNINDEX"].values)
        print(f"  PCA-ARDL CEV={info['cev']}: RMSE={ardl_m['RMSE']:.4f}")


if __name__ == "__main__":
    main()
