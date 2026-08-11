from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.ardl import ARDL

from .common import diagnostics_from_residuals, mape, paired_valid, rmse, rolling_one_step_forecast


def _load_ardl_config(project_root: Path) -> dict:
    config_path = project_root / "configs" / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("ardl", {})


# --- from step_06_select_and_forecast.py ---
def run_select_and_forecast(context: dict) -> dict:
    """P0-3A final phase + P0-X: ONE final ARDL refit on unique Train+Val,
    then a fixed-model rolling one-step-ahead forecast over ALL Test dates.

    Test is used here for the first time in the ARDL lifecycle — never
    during the sweep/selection phase (core/sweep.py::run_sweep_ardl).
    """
    project_root = context.get("PROJECT_ROOT", Path(__file__).resolve().parents[3])
    ardl_cfg = _load_ardl_config(project_root)
    use_ensemble = ardl_cfg.get("use_ensemble", False)
    ensemble_top_n = int(ardl_cfg.get("ensemble_top_n", 3))
    ensemble_criterion = ardl_cfg.get("ensemble_criterion", "RMSE_val")
    causal = bool(ardl_cfg.get("causal", True))
    hold_back = int(ardl_cfg.get("hold_back", 5))

    selected_pair = context.get("SELECTED_PAIR", (5, 2))
    if selected_pair not in context["ardl_results_by_pair"]:
        raise KeyError(f"Selected pair {selected_pair} was not fit successfully in the sweep.")

    y_train    = context["y_train"]
    X_train    = context["X_train"]
    y_val      = context["y_val"]
    X_val      = context["X_val"]
    y_trainval = context["y_trainval"]
    X_trainval = context["X_trainval"]
    y_test     = context["y_test"]
    X_test     = context["X_test"]

    # ── ONE final refit on unique Train+Val (no duplicate dates: trainval
    # is Train concatenated with Val, both already disjoint per P0-1) ─────
    p, q = selected_pair
    ardl_model = ARDL(
        endog=y_trainval, lags=p, exog=X_trainval, order=q,
        causal=causal, trend="c", hold_back=hold_back,
    )
    ardl_res = ardl_model.fit()

    # y_full/X_full must span Train+Val's tail (for lag history) through
    # Test so the first Test target's lags resolve to actual Train+Val
    # observations, per P0-X ("For the first Val/Test target, required lag
    # history must come from the preceding split").
    y_full_for_test = pd.concat([y_trainval, y_test]).sort_index()
    X_full_for_test = pd.concat([X_trainval, X_test]).sort_index()
    pred_test = rolling_one_step_forecast(ardl_res, y_full_for_test, X_full_for_test, y_test.index)

    # In-sample fitted values on Train+Val (diagnostic only — not a
    # selection metric, computed after selection is already final).
    pred_trainval = ardl_res.fittedvalues

    y_trainval_eval, pred_trainval_eval = paired_valid(y_trainval, ardl_res.fittedvalues)
    y_test_eval, pred_test_eval = paired_valid(y_test, pred_test)

    metrics = {
        "RMSE_trainval": rmse(y_trainval_eval, pred_trainval_eval),
        "MAE_trainval": float(mean_absolute_error(y_trainval_eval, pred_trainval_eval)),
        "MAPE_trainval(%)": mape(y_trainval_eval, pred_trainval_eval),
        "R2_trainval": float(r2_score(y_trainval_eval, pred_trainval_eval)),
        "RMSE_test": rmse(y_test_eval, pred_test_eval),
        "MAE_test": float(mean_absolute_error(y_test_eval, pred_test_eval)),
        "MAPE_test(%)": mape(y_test_eval, pred_test_eval),
        "R2_test": float(r2_score(y_test_eval, pred_test_eval)),
    }

    # ── Optional: Top-N ensemble ──────────────────────────────────────────
    # Each top candidate is refit on Train+Val (fresh, since the sweep phase
    # only fits Train-only per P0-3A) and forecast via the same rolling
    # one-step-ahead helper, so the ensemble respects the same information
    # boundary as the single selected model.
    ensemble_pred_test = None
    ensemble_metrics = None

    if use_ensemble and "ardl_sweep_table" in context:
        sweep = context["ardl_sweep_table"]
        sweep_ok = sweep[sweep["Status"] == "OK"].copy()

        crit = ensemble_criterion if ensemble_criterion in sweep_ok.columns else "RMSE_val"
        if crit not in sweep_ok.columns:
            crit = "RMSE_val"

        top_rows = sweep_ok.nsmallest(ensemble_top_n, crit)
        top_pairs = [(int(r["P"]), int(r["Q"])) for _, r in top_rows.iterrows()
                     if (int(r["P"]), int(r["Q"])) in context["ardl_results_by_pair"]]

        if len(top_pairs) >= 2:
            preds = []
            for pair in top_pairs:
                tp, tq = pair
                ens_model = ARDL(
                    endog=y_trainval, lags=tp, exog=X_trainval, order=tq,
                    causal=causal, trend="c", hold_back=hold_back,
                )
                ens_res = ens_model.fit()
                p_test = rolling_one_step_forecast(ens_res, y_full_for_test, X_full_for_test, y_test.index)
                preds.append(p_test.values)
                print(f"  [Ensemble] Including ARDL{pair} (refit on Train+Val, rolling one-step Test forecast)")

            ensemble_arr = np.mean(preds, axis=0)
            ensemble_pred_test = pd.Series(ensemble_arr, index=y_test.index)

            y_ens_eval, p_ens_eval = paired_valid(y_test, ensemble_pred_test)
            ensemble_metrics = {
                "Ensemble_RMSE_test": rmse(y_ens_eval, p_ens_eval),
                "Ensemble_MAE_test": float(mean_absolute_error(y_ens_eval, p_ens_eval)),
                "Ensemble_MAPE_test(%)": mape(y_ens_eval, p_ens_eval),
                "Ensemble_R2_test": float(r2_score(y_ens_eval, p_ens_eval)),
                "Ensemble_pairs": str(top_pairs),
                "Ensemble_criterion": crit,
            }
            print(f"  [Ensemble] Top-{len(top_pairs)} by {crit}:")
            print(f"  [Ensemble] RMSE_test = {ensemble_metrics['Ensemble_RMSE_test']:.6f}"
                  f"  (vs single {selected_pair}: {metrics['RMSE_test']:.6f})")

    # ── Save forecast tables ───────────────────────────────────────────────
    forecast_table = pd.DataFrame({
        "Date": y_test.index,
        "Actual_VNINDEX": y_test.values,
        "Predicted_VNINDEX": pred_test.values,
        "Residual": y_test.values - pred_test.values,
    })

    results_dir = context["PROJECT_ROOT"] / "outputs" / "ardl_vnindex_forecast"
    results_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = results_dir / f"ardl_test_forecast_P{selected_pair[0]}_Q{selected_pair[1]}.csv"
    forecast_table.to_csv(forecast_path, index=False)

    diag = diagnostics_from_residuals(ardl_res.resid)

    context.update({
        "SELECTED_PAIR": selected_pair,
        "ardl_model": ardl_model,
        "ardl_res": ardl_res,
        "pred_trainval": pred_trainval,
        "pred_test": pred_test,
        "metrics": metrics,
        "forecast_table": forecast_table,
        "forecast_path": forecast_path,
        "diag": diag,
        "ensemble_pred_test": ensemble_pred_test,
        "ensemble_metrics": ensemble_metrics,
    })

    print("ARDL step 6: selected pair", selected_pair)
    print("  RMSE_test =", f"{metrics['RMSE_test']:.6f}")
    print("  MAE_test  =", f"{metrics['MAE_test']:.6f}")
    print("  MAPE_test =", f"{metrics['MAPE_test(%)']:.6f}")
    print("  R2_test   =", f"{metrics['R2_test']:.6f}")
    print("  Forecast  =", forecast_path)

    # Chapter 4 CSV
    chapter4_df = pd.DataFrame({
        "Date": y_test.index,
        "Actual_VNINDEX": y_test.values,
        "Predicted_VNINDEX": pred_test.values,
        "Residual": y_test.values - pred_test.values,
        "APE_%": np.abs((y_test.values - pred_test.values) / (y_test.values + 1e-8)) * 100,
    })
    if ensemble_pred_test is not None:
        chapter4_df["Ensemble_Predicted"] = ensemble_pred_test.values
        chapter4_df["Ensemble_Residual"] = y_test.values - ensemble_pred_test.values

    chapter4_path = results_dir / "chapter4_ardl_forecast.csv"
    chapter4_df.to_csv(chapter4_path, index=False)
    print(f"Chapter 4 data saved to: {chapter4_path}")
    context["chapter4_csv_path"] = chapter4_path
    return context
