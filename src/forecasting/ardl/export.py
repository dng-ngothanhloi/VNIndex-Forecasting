from __future__ import annotations

import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

from .common import load_inputs


# --- from step_07_export_pkl.py ---
def run_export_pkl(context: dict) -> dict:
    export_dir = context["PROJECT_ROOT"] / "outputs" / "ardl_vnindex_forecast"
    export_dir.mkdir(parents=True, exist_ok=True)
    selected_pair = context["SELECTED_PAIR"]
    model_pkl_path = export_dir / f"ardl_model_P{selected_pair[0]}_Q{selected_pair[1]}.pkl"

    model_bundle = {
        "model_type": "statsmodels_ARDL",
        "selected_pair": selected_pair,
        "model_class": type(context["ardl_model"]).__name__,
        "results_class": type(context["ardl_res"]).__name__,
        "p_lags": list(context["ardl_model"]._lags),
        "q_lags_map": {k: list(v) for k, v in context["ardl_model"]._order.items()},
        "feature_columns": context["pc_cols"],
        "target_col": context["target_col"],
        "metrics": context["metrics"],
        "diagnostics": context["diag"],
        "forecast_path": str(context["forecast_path"]),
        "forecast_table_head": context["forecast_table"].head(10).to_dict(orient="records"),
        "model": context["ardl_model"],
        "results": context["ardl_res"],
        "exported_at": datetime.now().isoformat(),
        "version": "1.0",
    }

    with open(model_pkl_path, "wb") as f:
        pickle.dump(model_bundle, f)

    with open(model_pkl_path, "rb") as f:
        loaded_bundle = pickle.load(f)

    context.update({"model_pkl_path": model_pkl_path, "model_bundle": model_bundle, "loaded_bundle": loaded_bundle})
    print("ARDL step 7: saved pickle ->", model_pkl_path)
    print("  loaded pair:", loaded_bundle.get("selected_pair"))

    # export numeric tables into outputs/ardl_vnindex_report
    report_dir = context["PROJECT_ROOT"] / "outputs" / "ardl_vnindex_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # coefficients
    try:
        res = context["ardl_res"]
        coeffs = pd.DataFrame({
            "coef": res.params,
            "std_err": res.bse,
            "t": res.tvalues,
            "pvalue": res.pvalues,
        })
        coeffs.to_csv(report_dir / "ardl_coefficients.csv")
    except Exception as e:
        (report_dir / "ardl_coefficients_error.txt").write_text(str(e))

    # meta
    try:
        meta = {
            "aic": float(getattr(res, "aic", None)),
            "bic": float(getattr(res, "bic", None)),
            "hqic": float(getattr(res, "hqic", None)),
            "nobs": int(getattr(res, "nobs", len(res.model.endog))),
            "num_params": int(len(res.params)),
            "hold_back": int(getattr(res.model, "hold_back", getattr(res.model, "_hold_back", 0))),
        }
        (report_dir / "ardl_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    except Exception as e:
        (report_dir / "ardl_meta_error.txt").write_text(str(e))

    # diagnostics
    try:
        res = context.get("ardl_res")
        resid = getattr(res, "resid", None)
        hb = int(getattr(res.model, "hold_back", getattr(res.model, "_hold_back", 0)))
        dw = float(durbin_watson(resid)) if resid is not None else None
        # Ljung-Box at lag 10
        try:
            lb_df = acorr_ljungbox(resid, lags=[10], return_df=True)
            lb_stat = float(lb_df.iloc[0]["lb_stat"])
            lb_p = float(lb_df.iloc[0]["lb_pvalue"])
        except Exception:
            lb_stat = None
            lb_p = None

        # Breusch-Pagan (align by hold_back)
        try:
            bp = het_breuschpagan(resid[hb:], res.model.exog[hb:])
            bp_dict = {"lm": float(bp[0]), "lm_pvalue": float(bp[1]), "f": float(bp[2]), "f_pvalue": float(bp[3])}
        except Exception:
            bp_dict = {"error": "could not run breusch-pagan (shape misalign)"}

        diagnostics = {"durbin_watson": dw, "ljungbox_q10": lb_stat, "ljungbox_p_q10": lb_p, "breusch_pagan": bp_dict}
        (report_dir / "ardl_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    except Exception as e:
        (report_dir / "ardl_diagnostics_error.txt").write_text(str(e))

    # metrics
    try:
        metrics = context.get("metrics", {})
        (report_dir / "metrics_by_split.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    except Exception as e:
        (report_dir / "metrics_error.txt").write_text(str(e))

    # sweep summary (compute adjusted R2 if possible)
    try:
        sweep_path = context["PROJECT_ROOT"] / "outputs" / "ardl_vnindex_pca_sweep" / "sweep_results.csv"
        if sweep_path.exists():
            sweep = pd.read_csv(sweep_path)
            if {'trainval_r2', 'trainval_n', 'n_params'}.issubset(sweep.columns):
                def _adj_r2(row):
                    r2 = row['trainval_r2']
                    n = row['trainval_n']
                    p = row['n_params']
                    return 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else np.nan
                sweep['adj_r2'] = sweep.apply(_adj_r2, axis=1)
            sweep.to_csv(report_dir / 'sweep_summary.csv', index=False)
    except Exception as e:
        (report_dir / "sweep_error.txt").write_text(str(e))

    # predicted vs actual and copy existing forecast (add APE_%)
    try:
        if "forecast_table" in context and hasattr(context["forecast_table"], "to_csv"):
            ft = context["forecast_table"].copy()
            # compute APE_% safely
            if "Actual_VNINDEX" in ft.columns and "Predicted_VNINDEX" in ft.columns:
                eps = 1e-8
                ft["APE_%"] = (ft["Actual_VNINDEX"].replace(0, np.nan) - ft["Predicted_VNINDEX"]) .abs() / ft["Actual_VNINDEX"].replace(0, np.nan) * 100.0
                ft["APE_%"] = ft["APE_%"].fillna(0.0)
            ft.to_csv(report_dir / f"predicted_vs_actual_test_P{selected_pair[0]}_Q{selected_pair[1]}.csv", index=False)
        if "forecast_path" in context and Path(context["forecast_path"]).exists():
            # also copy existing forecast CSV
            shutil.copy(context["forecast_path"], report_dir / Path(context["forecast_path"]).name)
    except Exception as e:
        (report_dir / "predicted_export_error.txt").write_text(str(e))

    # OLS-equivalent metrics (R2, adj R2, F-stat)
    try:
        y_aligned = res.model.endog[hb:]
        X_aligned = res.model.exog[hb:]
        ols = sm.OLS(y_aligned, X_aligned).fit()
        ols_meta = {"rsquared": float(ols.rsquared), "rsquared_adj": float(ols.rsquared_adj), "fvalue": float(ols.fvalue), "f_pvalue": float(ols.f_pvalue)}
        (report_dir / "ardl_ols_metrics.json").write_text(json.dumps(ols_meta, indent=2, ensure_ascii=False))
    except Exception as e:
        (report_dir / "ardl_ols_metrics_error.txt").write_text(str(e))

    # ADF tests
    def _compute_adf(s):
        try:
            r = adfuller(pd.Series(s).dropna(), autolag='AIC')
            return {"adf_stat": float(r[0]), "pvalue": float(r[1]), "usedlag": int(r[2]), "nobs": int(r[3])}
        except Exception as e:
            return {"error": str(e)}

    try:
        inputs = load_inputs(context["PROJECT_ROOT"])
        pc_cols = inputs.get("pc_cols", [])[:11]
        vn_col = inputs.get("target_col", "VNINDEX")
        adf_results = {}
        vn_series = pd.concat([inputs["train_df"][vn_col], inputs["val_df"][vn_col], inputs["test_df"][vn_col]]).dropna()
        adf_results["VNINDEX_level"] = _compute_adf(vn_series)
        adf_results["VNINDEX_diff1"] = _compute_adf(vn_series.diff().dropna())
        for pc in pc_cols:
            series = pd.concat([inputs["train_df"][pc], inputs["val_df"][pc], inputs["test_df"][pc]]).dropna()
            adf_results[f"{pc}_level"] = _compute_adf(series)
            adf_results[f"{pc}_diff1"] = _compute_adf(series.diff().dropna())
        (report_dir / "adf_results.json").write_text(json.dumps(adf_results, indent=2, ensure_ascii=False))
    except Exception as e:
        (report_dir / "adf_results_error.txt").write_text(str(e))

    context.update({"report_dir": report_dir})
    print("ARDL step 7: exports written to", report_dir)

    return context
