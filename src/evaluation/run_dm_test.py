"""
run_dm_test.py – Diebold-Mariano test: ARDL vs LSTM
=====================================================
Run after both ARDL and LSTM pipelines complete.

Usage:
    python experiments/run_dm_test.py
    python experiments/run_dm_test.py --cev 0.90   # for multi-CEV results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# src/evaluation/run_dm_test.py -> parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.dm_test import diebold_mariano_test
from src.evaluation.metrics import regression_metrics


def load_forecasts(cev: float | None = None) -> pd.DataFrame:
    """
    Load ARDL and LSTM test-set forecasts and merge on Date.

    Parameters
    ----------
    cev : float or None
        If provided, load from outputs/cev_{cev:.2f}/...
        Otherwise load from default outputs/...
    """
    if cev is not None:
        base = PROJECT_ROOT / "outputs" / f"cev_{cev:.2f}"
    else:
        base = PROJECT_ROOT / "outputs"

    # ── ARDL forecast ─────────────────────────────────────────
    ardl_candidates = [
        base / "ardl_vnindex_forecast" / "chapter4_ardl_forecast.csv",
    ]
    # Also look for any ardl_test_forecast_P*_Q*.csv
    ardl_dir = base / "ardl_vnindex_forecast"
    if ardl_dir.exists():
        ardl_candidates += sorted(ardl_dir.glob("ardl_test_forecast_P*.csv"))

    ardl_df = None
    for p in ardl_candidates:
        if p.exists():
            ardl_df = pd.read_csv(p, parse_dates=["Date"])
            print(f"[DM] ARDL forecast: {p.name}  ({len(ardl_df)} rows)")
            break

    if ardl_df is None:
        raise FileNotFoundError(
            f"No ARDL forecast CSV found in {ardl_dir}. "
            "Run experiments/run_ardl_experiment.py first."
        )

    # ── LSTM forecast ──────────────────────────────────────────
    lstm_dir = base / "lstm_vnindex_sweep"
    lstm_df = None

    # Try sweep_summary to find best Val_RMSE model
    summary_path = lstm_dir / "sweep_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        best_row = summary.loc[summary["Val_RMSE"].idxmin()]
        lb = int(best_row["Lookback"])
        bs = int(best_row["Batch_size"])
        best_file = lstm_dir / f"predictions_lookback_{lb}_batch_{bs}.csv"
        if best_file.exists():
            lstm_df = pd.read_csv(best_file, parse_dates=["Date"])
            print(f"[DM] LSTM forecast: {best_file.name}  ({len(lstm_df)} rows, best Val_RMSE)")

    # Fallback: any predictions_lookback_*.csv
    if lstm_df is None:
        for f in sorted(lstm_dir.glob("predictions_lookback_*.csv")):
            lstm_df = pd.read_csv(f, parse_dates=["Date"])
            print(f"[DM] LSTM forecast (fallback): {f.name}  ({len(lstm_df)} rows)")
            break

    if lstm_df is None:
        raise FileNotFoundError(
            f"No LSTM prediction CSV found in {lstm_dir}. "
            "Run experiments/run_lstm_experiment.py first."
        )

    # ── P0-5: FAIL-FAST same-population checks (never silently drop rows) ──
    # ARDL and LSTM must predict the SAME Test target dates, SAME y_true,
    # SAME count -- an inner join that silently drops mismatched rows would
    # hide a scientific defect (e.g. one model's Test population being a
    # different size/date range than the other's), so we assert BEFORE
    # merging, then assert again AFTER merging that zero rows were dropped.
    ardl_dates = pd.Index(ardl_df["Date"])
    lstm_dates = pd.Index(lstm_df["Date"])

    if len(ardl_df) != len(lstm_df):
        raise AssertionError(
            f"[DM][FAIL-FAST] ARDL Test population (n={len(ardl_df)}) != "
            f"LSTM Test population (n={len(lstm_df)}). P0-5 requires both "
            f"models to forecast the SAME Test target set. Refusing to "
            f"silently inner-join a mismatched population."
        )

    if not ardl_dates.equals(lstm_dates):
        only_in_ardl = ardl_dates.difference(lstm_dates)
        only_in_lstm = lstm_dates.difference(ardl_dates)
        raise AssertionError(
            f"[DM][FAIL-FAST] ARDL and LSTM Test dates differ. "
            f"Dates only in ARDL: {list(only_in_ardl[:5])}{'...' if len(only_in_ardl) > 5 else ''} "
            f"Dates only in LSTM: {list(only_in_lstm[:5])}{'...' if len(only_in_lstm) > 5 else ''}. "
            f"P0-5 requires identical Test target dates for both models."
        )

    # ── Merge on Date (safe now: dates are already verified identical and
    # unique per side, so this is a 1:1 alignment, never a lossy inner join). ──
    merged = pd.merge(
        ardl_df[["Date", "Actual_VNINDEX", "Predicted_VNINDEX"]].rename(
            columns={"Predicted_VNINDEX": "ARDL_Pred"}
        ),
        lstm_df[["Date", "Actual_VNINDEX", "Predicted_VNINDEX"]].rename(
            columns={"Actual_VNINDEX": "LSTM_Actual_VNINDEX", "Predicted_VNINDEX": "LSTM_Pred"}
        ),
        on="Date",
        how="inner",
    )

    # ── Post-merge fail-fast: prove ZERO rows were dropped and y_true matches ──
    if len(merged) != len(ardl_df) or len(merged) != len(lstm_df):
        raise AssertionError(
            f"[DM][FAIL-FAST] Merge dropped rows: ARDL n={len(ardl_df)}, "
            f"LSTM n={len(lstm_df)}, merged n={len(merged)}. Expected all "
            f"three counts to be equal (P0-5: same Test population)."
        )

    y_true_mismatch = ~np.isclose(merged["Actual_VNINDEX"], merged["LSTM_Actual_VNINDEX"], equal_nan=True)
    if y_true_mismatch.any():
        bad_dates = merged.loc[y_true_mismatch, "Date"].tolist()
        raise AssertionError(
            f"[DM][FAIL-FAST] ARDL Actual_VNINDEX != LSTM Actual_VNINDEX for "
            f"{y_true_mismatch.sum()} date(s) (e.g. {bad_dates[:5]}). Both "
            f"models must be evaluated against the SAME y_true (P0-5)."
        )
    merged = merged.drop(columns=["LSTM_Actual_VNINDEX"])

    print(f"[DM] Same-population check PASSED: n={len(merged)}, dates identical, y_true identical.")
    print(f"[DM] Merged: {len(merged)} common observations")
    print(f"[DM] Date range: {merged['Date'].min().date()} → {merged['Date'].max().date()}")
    return merged


def run_dm_test(cev: float | None = None) -> dict:
    print("=" * 72)
    title = f"DIEBOLD-MARIANO TEST: ARDL vs LSTM"
    if cev is not None:
        title += f"  (CEV={cev:.2f})"
    print(title)
    print("=" * 72)

    df = load_forecasts(cev)

    actual    = df["Actual_VNINDEX"].values
    ardl_pred = df["ARDL_Pred"].values
    lstm_pred = df["LSTM_Pred"].values

    # ── P0-5: standalone metrics recomputed on this SAME (already
    # fail-fast-verified) population, for both models, before DM. ─────────
    standalone = {
        "ARDL": regression_metrics(actual, ardl_pred),
        "LSTM": regression_metrics(actual, lstm_pred),
        "n": int(len(df)),
    }
    print("\n--- Standalone Test Metrics (same population, n={}) ---".format(standalone["n"]))
    for model_name in ("ARDL", "LSTM"):
        m = standalone[model_name]
        print(f"  {model_name:5s} | RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  MAPE={m['MAPE(%)']:.4f}%  R2={m['R2']:.4f}")

    results = {}
    for loss in ("mse", "mae"):
        dm = diebold_mariano_test(
            actual, ardl_pred, lstm_pred,
            loss_type=loss,
            alternative="two-sided",
        )
        results[loss] = dm
        print(f"\n--- DM Test ({loss.upper()}) ---")
        print(f"  DM Statistic : {dm['dm_stat']:+.4f}")
        print(f"  p-value      : {dm['p_value']:.4f}")
        print(f"  Mean diff    : {dm['mean_diff']:.4f}  (negative → ARDL better)")
        print(f"  SE diff      : {dm['se_diff']:.4f}")
        print(f"  n={dm['sample_size']}, max_lag={dm['max_lag']}")
        print(f"  → {dm['conclusion']}")

    # ── Save outputs ───────────────────────────────────────────
    if cev is not None:
        out_dir = PROJECT_ROOT / "outputs" / f"cev_{cev:.2f}" / "model_comparison"
    else:
        out_dir = PROJECT_ROOT / "outputs" / "model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_rows = []
    for loss_type, dm in results.items():
        results_rows.append({
            "Loss_Type":   loss_type.upper(),
            "DM_Stat":     dm["dm_stat"],
            "p_value":     dm["p_value"],
            "Mean_Diff":   dm["mean_diff"],
            "SE_Diff":     dm["se_diff"],
            "Sample_Size": dm["sample_size"],
            "Significant": dm["significant"],
            "Conclusion":  dm["conclusion"],
        })

    results_df = pd.DataFrame(results_rows)
    csv_path = out_dir / "dm_test_results.csv"
    results_df.to_csv(csv_path, index=False)

    # ── P0-5: persist standalone same-population metrics alongside DM ──────
    standalone_rows = []
    for model_name in ("ARDL", "LSTM"):
        m = standalone[model_name]
        standalone_rows.append({
            "Model": model_name, "N": standalone["n"],
            "RMSE": m["RMSE"], "MAE": m["MAE"], "MAPE(%)": m["MAPE(%)"], "R2": m["R2"],
        })
    standalone_df = pd.DataFrame(standalone_rows)
    standalone_csv_path = out_dir / "standalone_test_metrics.csv"
    standalone_df.to_csv(standalone_csv_path, index=False)
    print(f"[COMP] Standalone metrics saved: {standalone_csv_path}")

    report_path = out_dir / "dm_test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DIEBOLD-MARIANO TEST REPORT\n")
        f.write("=" * 72 + "\n\n")
        if cev is not None:
            f.write(f"CEV threshold: {cev:.2f}\n")
        f.write(f"Date range:    {df['Date'].min().date()} → {df['Date'].max().date()}\n")
        f.write(f"Sample size:   {len(df)} observations\n")
        f.write(f"Forecast horizon: T+1 (h=1, same for both models, P0-5)\n\n")

        f.write("STANDALONE TEST METRICS (same population, P0-5):\n")
        for model_name in ("ARDL", "LSTM"):
            m = standalone[model_name]
            f.write(f"  {model_name}: RMSE={m['RMSE']:.6f}  MAE={m['MAE']:.6f}  "
                    f"MAPE={m['MAPE(%)']:.6f}%  R2={m['R2']:.6f}\n")
        f.write("\n")

        for loss_type, dm in results.items():
            f.write(f"{loss_type.upper()} LOSS:\n")
            f.write(f"  DM Statistic : {dm['dm_stat']:+.6f}\n")
            f.write(f"  p-value      : {dm['p_value']:.6f}\n")
            f.write(f"  Mean diff    : {dm['mean_diff']:.6f}\n")
            f.write(f"  Conclusion   : {dm['conclusion']}\n\n")

        dm_mse = results["mse"]
        dm_mae = results["mae"]
        lower_mse_model = "ARDL" if dm_mse["mean_diff"] < 0 else "LSTM"
        lower_mae_model = "ARDL" if dm_mae["mean_diff"] < 0 else "LSTM"

        f.write("INTERPRETATION:\n")
        f.write(
            "  NOTE: 'has lower observed loss' describes the sample MSE/MAE on this\n"
            "  Test set only. It is a statistically defensible claim ONLY when paired\n"
            "  with the DM significance verdict below -- observing a lower point\n"
            "  estimate is not itself evidence of a real accuracy difference.\n\n"
        )

        # MSE: distinguish "observed lower loss" from "statistically significant"
        f.write(f"  MSE  : {lower_mse_model} has the lower observed MSE loss on this Test set "
                f"(diff={dm_mse['mean_diff']:.4f}).\n")
        if dm_mse["p_value"] < 0.05:
            f.write(f"         This difference IS statistically significant at the 5% level "
                    f"(p={dm_mse['p_value']:.4f}).\n")
        elif dm_mse["p_value"] < 0.10:
            f.write(f"         This difference is marginal -- significant at the 10% level but "
                    f"NOT significant at the conventional 5% level (p={dm_mse['p_value']:.4f}). "
                    f"Do not report this as a confirmed accuracy difference.\n")
        else:
            f.write(f"         This difference is NOT statistically significant "
                    f"(p={dm_mse['p_value']:.4f}).\n")

        # MAE: same distinction, independently evaluated
        f.write(f"\n  MAE  : {lower_mae_model} has the lower observed MAE loss on this Test set "
                f"(diff={dm_mae['mean_diff']:.4f}).\n")
        if dm_mae["p_value"] < 0.05:
            f.write(f"         This difference IS statistically significant in favor of "
                    f"{lower_mae_model} (p={dm_mae['p_value']:.4f}).\n")
        elif dm_mae["p_value"] < 0.10:
            f.write(f"         This difference is marginal -- significant at the 10% level but "
                    f"NOT at the 5% level (p={dm_mae['p_value']:.4f}).\n")
        else:
            f.write(f"         This difference is NOT statistically significant "
                    f"(p={dm_mae['p_value']:.4f}).\n")

    print(f"\n[COMP] Results saved: {csv_path}")
    print(f"[COMP] Report saved:  {report_path}")
    print("=" * 72)

    results["standalone_metrics"] = standalone
    return results


def main():
    parser = argparse.ArgumentParser(description="Diebold-Mariano test: ARDL vs LSTM")
    parser.add_argument("--cev", type=float, default=None,
                        help="CEV threshold (e.g. 0.90) for multi-CEV results")
    args = parser.parse_args()
    run_dm_test(cev=args.cev)


if __name__ == "__main__":
    main()
