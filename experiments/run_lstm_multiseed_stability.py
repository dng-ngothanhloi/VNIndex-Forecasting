#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_lstm_multiseed_stability.py – Phase 3D: LSTM multi-seed stability
===========================================================================
Scientific stability evaluation (statistical-validity policy: multi-seed
execution with seeds [42, 52, 62, 72, 82], reported as mean +/- std) --
NOT another architecture refactor and NOT a hyperparameter search.

Protocol (frozen a priori, per the pipeline-corrections task's Stage 3):
  1. Run the LSTM Train->Val tuning sweep EXACTLY ONCE, with seed=42 as
     the PREDECLARED reference tuning run (reuses
     src/forecasting/lstm/sweep.py::run_train_and_evaluate unmodified --
     no new tuning logic).
  2. Freeze the selected (lookback, batch_size, best_epoch) from that
     seed=42 run. This selection is NEVER repeated or re-derived per
     seed, and Test is NEVER used to make this selection (already
     guaranteed by run_train_and_evaluate's P0-3B contract).
  3. For each seed in [42, 52, 62, 72, 82]: instantiate a NEW model with
     that seed and refit on the SAME frozen (lookback, batch_size,
     best_epoch) via the SAME shared
     src/forecasting/lstm/sweep.py::final_refit_and_forecast helper --
     only the random weight initialization/training stochasticity
     differs across seeds. Persist per-seed predictions/metrics.
  4. Compute mean/std/min/max Test RMSE/MAE/MAPE across the 5 seeds.
  5. Optionally run the DM comparison against ARDL for EACH seed as a
     robustness diagnostic (reported per-seed, never averaged/selected).

Usage:
    python experiments/run_lstm_multiseed_stability.py
    python experiments/run_lstm_multiseed_stability.py --skip-dm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.lstm.setup import run_imports, run_paths
from src.forecasting.lstm.data import run_load_data, run_prepare_data
from src.forecasting.lstm.sweep import run_train_and_evaluate, final_refit_and_forecast
from src.evaluation.metrics import regression_metrics

GOVERNED_SEEDS = [42, 52, 62, 72, 82]
REFERENCE_SEED = 42  # predeclared a priori -- must NOT be chosen post-hoc by Test performance


def _hdr(text: str, width: int = 78) -> None:
    print("\n" + "=" * width)
    print(f" {text} ".center(width))
    print("=" * width)


def run_reference_tuning() -> dict:
    """Run the tuning sweep exactly once with seed=42 (the fixed seed
    already hardcoded inside run_train_and_evaluate's build_model
    closure) -- reused UNMODIFIED. Returns the resulting context, which
    contains the frozen selected_lookback/selected_batch_size/
    final_refit_epochs plus the already-fitted reference-seed final model
    and its Test predictions (seed=42's own row in the stability table)."""
    _hdr(f"REFERENCE TUNING RUN (seed={REFERENCE_SEED}, predeclared)")
    context: dict = {}
    context = run_imports(context)
    context = run_paths(context)
    context = run_load_data(context)
    context = run_prepare_data(context)
    context = run_train_and_evaluate(context)
    return context


def run_stability_seeds(context: dict) -> pd.DataFrame:
    """For each seed in GOVERNED_SEEDS, refit a NEW model on the frozen
    (lookback, batch_size, best_epoch) selected by the reference tuning
    run, using the SAME shared final_refit_and_forecast helper. Persists
    per-seed predictions and returns a summary DataFrame."""
    PROJECT_ROOT_ctx = context["PROJECT_ROOT"]
    config_path = PROJECT_ROOT_ctx / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    lstm_cfg = config.get("lstm", {})

    selected_lookback = context["selected_lookback"]
    selected_batch_size = context["selected_batch_size"]
    best_epoch = context["final_refit_epochs"]

    results_dir = PROJECT_ROOT_ctx / "outputs" / "lstm_vnindex_multiseed"
    results_dir.mkdir(parents=True, exist_ok=True)

    _hdr(f"MULTI-SEED STABILITY: lookback={selected_lookback} batch={selected_batch_size} "
         f"epochs={best_epoch} (FROZEN from seed={REFERENCE_SEED} reference tuning)")

    rows = []
    test_dates_ref = None

    for seed in GOVERNED_SEEDS:
        print(f"\n[SEED {seed}] Refitting NEW model (frozen hyperparameters, seed varies weight init only)...")
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
            learning_rate=lstm_cfg.get("learning_rate", 1e-4),
            lstm_units=lstm_cfg.get("lstm_units", [64, 32]),
            dense_units=lstm_cfg.get("dense_units", [16]),
            dropout_rate=lstm_cfg.get("dropout_rate", 0.2),
            use_batch_norm=lstm_cfg.get("use_batch_norm", False),
            seed=seed,
        )

        test_dates = result["test_dates"]
        if test_dates_ref is None:
            test_dates_ref = test_dates
        else:
            if not test_dates.equals(test_dates_ref):
                raise AssertionError(
                    f"[Phase 3D][FAIL-FAST] seed={seed} produced different Test "
                    f"dates than the reference seed={REFERENCE_SEED} run -- every "
                    f"seed must forecast the SAME Test population."
                )

        pred_table = pd.DataFrame({
            "Date": test_dates,
            "Actual_VNINDEX": result["y_test_true"],
            "Predicted_VNINDEX": result["test_pred"],
            "Residual": result["y_test_true"] - result["test_pred"],
        })
        pred_path = results_dir / f"predictions_seed_{seed}.csv"
        pred_table.to_csv(pred_path, index=False)

        m = regression_metrics(result["y_test_true"], result["test_pred"])
        row = {"Seed": seed, "N_Test": len(test_dates), **m}
        rows.append(row)
        print(f"[SEED {seed}] Test RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} MAPE={m['MAPE(%)']:.4f}% R2={m['R2']:.4f}")

        # Free the TF model/graph immediately after each seed's refit --
        # never accumulate 5 model objects in RAM (same discipline as P1).
        del result
        import tensorflow as tf
        tf.keras.backend.clear_session()

    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "multiseed_test_metrics.csv", index=False)

    agg_rows = []
    for metric in ("RMSE", "MAE", "MAPE(%)", "R2"):
        agg_rows.append({
            "Metric": metric,
            "Mean": summary[metric].mean(),
            "Std": summary[metric].std(ddof=1),
            "Min": summary[metric].min(),
            "Max": summary[metric].max(),
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(results_dir / "multiseed_summary_stats.csv", index=False)

    print("\n[Phase 3D] Per-seed Test metrics:")
    print(summary.to_string(index=False))
    print("\n[Phase 3D] Mean +/- Std summary (n=5 seeds):")
    for _, r in agg_df.iterrows():
        print(f"  {r['Metric']:8s}: {r['Mean']:.4f} +/- {r['Std']:.4f}  (min={r['Min']:.4f}, max={r['Max']:.4f})")

    print(f"\n[SAVED] {results_dir / 'multiseed_test_metrics.csv'}")
    print(f"[SAVED] {results_dir / 'multiseed_summary_stats.csv'}")

    return summary


def run_per_seed_dm_diagnostic(summary: pd.DataFrame, context: dict) -> None:
    """OPTIONAL robustness diagnostic: run the DM test (ARDL vs LSTM)
    once per seed, reporting each seed's result separately. Never
    averages p-values, never selects the seed with the best DM result,
    and never constructs a seed ensemble -- purely diagnostic per
    statistical-validity policy and the Stage 3 spec's explicit
    forbidden-practices list."""
    from src.evaluation.dm_test import diebold_mariano_test

    PROJECT_ROOT_ctx = context["PROJECT_ROOT"]
    ardl_dir = PROJECT_ROOT_ctx / "outputs" / "ardl_vnindex_forecast"
    ardl_candidates = sorted(ardl_dir.glob("ardl_test_forecast_P*.csv")) + [
        ardl_dir / "chapter4_ardl_forecast.csv"
    ]
    ardl_path = next((p for p in ardl_candidates if p.exists()), None)
    if ardl_path is None:
        print("\n[Phase 3D][DM diagnostic] SKIPPED: no ARDL forecast found "
              "(run experiments/run_ardl_experiment.py first).")
        return

    ardl_df = pd.read_csv(ardl_path, parse_dates=["Date"])
    multiseed_dir = PROJECT_ROOT_ctx / "outputs" / "lstm_vnindex_multiseed"

    _hdr("PER-SEED DM DIAGNOSTIC (ARDL vs LSTM, reported separately per seed)")
    rows = []
    for seed in GOVERNED_SEEDS:
        lstm_df = pd.read_csv(multiseed_dir / f"predictions_seed_{seed}.csv", parse_dates=["Date"])
        merged = pd.merge(
            ardl_df[["Date", "Actual_VNINDEX", "Predicted_VNINDEX"]].rename(columns={"Predicted_VNINDEX": "ARDL_Pred"}),
            lstm_df[["Date", "Predicted_VNINDEX"]].rename(columns={"Predicted_VNINDEX": "LSTM_Pred"}),
            on="Date", how="inner",
        )
        if len(merged) != len(ardl_df) or len(merged) != len(lstm_df):
            print(f"[SEED {seed}] SKIPPED DM diagnostic: population mismatch "
                  f"(ARDL n={len(ardl_df)}, LSTM n={len(lstm_df)}, merged n={len(merged)}).")
            continue

        dm = diebold_mariano_test(
            merged["Actual_VNINDEX"].values, merged["ARDL_Pred"].values, merged["LSTM_Pred"].values,
            loss_type="mse", alternative="two-sided",
        )
        rows.append({
            "Seed": seed, "N": dm["sample_size"], "DM_Stat": dm["dm_stat"],
            "p_value": dm["p_value"], "Significant": dm["significant"],
        })
        print(f"[SEED {seed}] DM(MSE) stat={dm['dm_stat']:+.4f} p={dm['p_value']:.4f} "
              f"significant={dm['significant']}")

    if rows:
        dm_df = pd.DataFrame(rows)
        out_path = multiseed_dir / "multiseed_dm_diagnostic.csv"
        dm_df.to_csv(out_path, index=False)
        print(f"\n[SAVED] {out_path}")
        print("[Phase 3D] NOTE: each seed's DM result is reported independently. "
              "No p-value averaging, no best-seed selection, no seed ensemble.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3D: LSTM multi-seed stability")
    p.add_argument("--skip-dm", action="store_true", help="Skip the per-seed DM diagnostic")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    context = run_reference_tuning()
    summary = run_stability_seeds(context)
    if not args.skip_dm:
        run_per_seed_dm_diagnostic(summary, context)
    _hdr("PHASE 3D COMPLETE")


if __name__ == "__main__":
    main()
