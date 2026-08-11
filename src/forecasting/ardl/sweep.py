from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.ardl import ARDL

from .common import mape, paired_valid, rmse, rolling_one_step_forecast


def _load_ardl_config(project_root: Path) -> dict:
    config_path = project_root / "configs" / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("ardl", {})


# --- from step_05_sweep_ardl.py ---
def select_best_model(
    sweep_table: pd.DataFrame,
    criterion: str = "BIC",
    criterion_threshold: float | None = None,
    rmse_val_threshold: float | None = None,
    require_status_ok: bool = True,
) -> tuple[int, int]:
    """
    Select best ARDL model from sweep results using a two-condition filter.

    Selection procedure
    --------------------
    1. Keep only models with Status == "OK".
    2. Condition 1 (information criterion cap): if `criterion` is one of
       AIC/BIC/HQIC and `criterion_threshold` is provided, keep only models
       whose criterion value is strictly below the threshold.
    3. Condition 2 (out-of-sample val guard): if `rmse_val_threshold` is
       provided, keep only models whose RMSE_val is strictly below it.
       This guards against overfit models that minimise the information
       criterion in-sample but generalise poorly on the validation window.
    4. Among the surviving candidates, pick the one that minimises
       `criterion` (lower = better for every supported criterion).

    If a filter would empty the candidate set, that filter is skipped with a
    warning so the pipeline never crashes on overly tight thresholds.

    Parameters
    ----------
    sweep_table : DataFrame
        ARDL sweep output with columns P, Q, Status, AIC, BIC, HQIC, RMSE_val,
        RMSE_test, etc.
    criterion : {"AIC", "BIC", "HQIC", "RMSE_val", "MAE_val",
                 "RMSE_test", "MAE_test", "RMSE_trainval", "MAE_trainval"}
        Metric to minimise among the filtered candidates.
    criterion_threshold : float, optional
        Upper bound for the information criterion (only used when `criterion`
        is AIC/BIC/HQIC). None disables this filter.
    rmse_val_threshold : float, optional
        Upper bound for RMSE_val. None disables this filter.
    require_status_ok : bool
        Only consider rows where Status == "OK".

    Returns
    -------
    (P, Q) tuple of the selected model.
    """
    valid_criteria = ["AIC", "BIC", "HQIC", "RMSE_val", "MAE_val", "RMSE_test", "MAE_test", "RMSE_trainval", "MAE_trainval"]
    if criterion not in valid_criteria:
        raise ValueError(f"criterion must be one of {valid_criteria}")

    df = sweep_table.copy()
    if require_status_ok:
        df = df[df["Status"] == "OK"]
    if len(df) == 0:
        raise ValueError("No valid models found in sweep results")

    candidates = df.copy()
    applied: list[str] = []

    # ── Condition 1: information-criterion cap ────────────────────────────
    if criterion in ("AIC", "BIC", "HQIC") and criterion_threshold is not None:
        filtered = candidates[candidates[criterion] < criterion_threshold]
        if len(filtered) > 0:
            candidates = filtered
            applied.append(f"{criterion} < {criterion_threshold}")
        else:
            print(f"[Auto-selection][WARN] No model with {criterion} < "
                  f"{criterion_threshold}; skipping this filter.")

    # ── Condition 2: out-of-sample validation guard ──────────────────────
    if rmse_val_threshold is not None and "RMSE_val" in candidates.columns:
        filtered = candidates[candidates["RMSE_val"] < rmse_val_threshold]
        if len(filtered) > 0:
            candidates = filtered
            applied.append(f"RMSE_val < {rmse_val_threshold}")
        else:
            print(f"[Auto-selection][WARN] No model with RMSE_val < "
                  f"{rmse_val_threshold}; skipping this filter.")

    # ── Pick lowest `criterion` among survivors ──────────────────────────
    best_idx = candidates[criterion].idxmin()
    best_row = candidates.loc[best_idx]
    p, q = int(best_row["P"]), int(best_row["Q"])

    print(f"[Auto-selection] Filters applied: {', '.join(applied) if applied else 'none'}")
    print(f"[Auto-selection] Candidates after filtering: {len(candidates)}/{len(df)}")
    print(f"[Auto-selection] Best model by {criterion}: ARDL({p},{q})")
    print(f"  {criterion:12s} = {best_row[criterion]:.6f}")
    if "RMSE_val" in best_row:
        print(f"  RMSE_val     = {best_row['RMSE_val']:.6f}")
    if "RMSE_test" in best_row and criterion != "RMSE_test":
        print(f"  RMSE_test    = {best_row['RMSE_test']:.6f}")

    return (p, q)


def run_sweep_ardl(context: dict) -> dict:
    """P0-2 + P0-3A + P0-X candidate lifecycle.

    For each (p, q) candidate:
      1. Fit ARDL on TRAIN ONLY, with causal=True (no PC.L0) and a FIXED
         hold_back across all candidates (fair IC comparison).
      2. BIC/AIC/HQIC come from that Train-only fit.
      3. RMSE_val/MAE_val/etc. come from a FIXED-MODEL rolling one-step-ahead
         forecast (see common.rolling_one_step_forecast / P0-X) over ALL Val
         dates, using ACTUAL observed history (never the model's own prior
         predictions, never PC(t)/y(t)).
    Test metrics are NOT computed here — Test is never touched during
    candidate selection (P0-3A). The final Train+Val refit + Test rolling
    forecast happens once, after selection, in core/forecast.py.
    """
    project_root = context.get("PROJECT_ROOT", Path(__file__).resolve().parents[3])
    ardl_cfg = _load_ardl_config(project_root)

    # Build pq_pairs from config (or defaults)
    p_values = ardl_cfg.get("p_values", [1, 2, 3, 4, 5])
    q_values = ardl_cfg.get("q_values", [1, 2, 3, 4, 5])
    causal = bool(ardl_cfg.get("causal", True))
    hold_back = int(ardl_cfg.get("hold_back", 5))

    # ── NoReduction high-dimensionality safety rule ──────────────────
    # When using raw features (reduction_method=none, typically F=318),
    # Q>1 would produce more exogenous coefficients than effective
    # observations, making estimation numerically inappropriate.
    # Restrict Q=[1] for NoReduction; PCA keeps full grid unchanged.
    config_path = project_root / "configs" / "config.yaml"
    _reduction_method = "pca"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as _f:
            _reduction_method = yaml.safe_load(_f).get("reduction", {}).get("method", "pca")

    n_features = len(context["pc_cols"])
    if _reduction_method == "none" and n_features > 50:
        q_values_effective = [1]
        print(f"[ARDL Sweep] NoReduction with {n_features} features detected — "
              f"restricting Q=[1] (safety rule: avoid parameter explosion)")
    else:
        q_values_effective = q_values

    pq_pairs = [(p, q) for p in p_values for q in q_values_effective]

    print(f"[ARDL Sweep] P={p_values}  Q={q_values_effective}  causal={causal}  hold_back={hold_back}  → {len(pq_pairs)} configs")

    # Keep train and val separate — Test is intentionally excluded from this
    # stage entirely (P0-3A: "Test metrics must NOT be computed during
    # candidate selection").
    train_df = context["train_df"]
    val_df   = context["val_df"]
    test_df  = context["test_df"]
    trainval_df = pd.concat([train_df, val_df], axis=0).sort_index()

    y_train    = train_df[context["target_col"]].astype(float)
    X_train    = train_df[context["pc_cols"]].astype(float)
    y_val      = val_df[context["target_col"]].astype(float)
    X_val      = val_df[context["pc_cols"]].astype(float)
    y_trainval = trainval_df[context["target_col"]].astype(float)
    X_trainval = trainval_df[context["pc_cols"]].astype(float)
    y_test     = test_df[context["target_col"]].astype(float)
    X_test     = test_df[context["pc_cols"]].astype(float)

    print("ARDL step 5: train period  ", y_train.index.min().date(), "->", y_train.index.max().date())
    print("ARDL step 5: val period    ", y_val.index.min().date(), "->", y_val.index.max().date())
    print("ARDL step 5: test period   ", y_test.index.min().date(), "->", y_test.index.max().date())

    sweep_dir = context["PROJECT_ROOT"] / "outputs" / "ardl_vnindex_pca_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    ardl_results_by_pair = {}
    sweep_rows = []

    for p, q in pq_pairs:
        row = {"P": p, "Q": q}
        try:
            # ── Fit on TRAIN ONLY (P0-3A: candidate BIC/selection must come
            # from a legitimate Train→Val evaluation, never Train+Val) ────
            model_train = ARDL(
                endog=y_train, lags=p, exog=X_train, order=q,
                causal=causal, trend="c", hold_back=hold_back,
            )
            res_train = model_train.fit()

            # ── Rolling one-step-ahead forecast over ALL Val dates (P0-X):
            # fixed coefficients from res_train, actual history spanning
            # Train's tail + Val (never the model's own predictions). ─────
            pred_val = rolling_one_step_forecast(
                res_train, y_trainval, X_trainval, y_val.index,
            )
            y_v_eval, y_v_pred = paired_valid(y_val, pred_val)

            row.update({
                "Status": "OK",
                "Num_Params": int(len(res_train.params)),
                "AIC": float(res_train.aic),
                "BIC": float(res_train.bic),
                "HQIC": float(res_train.hqic),
                # validation out-of-sample (Train-only fit, rolling one-step)
                "RMSE_val": rmse(y_v_eval, y_v_pred),
                "MAE_val": float(mean_absolute_error(y_v_eval, y_v_pred)),
                "MAPE_val(%)": mape(y_v_eval, y_v_pred),
                "R2_val": float(r2_score(y_v_eval, y_v_pred)),
            })

            ardl_results_by_pair[(p, q)] = {
                "model_train": model_train,
                "res_train": res_train,
                "pred_val": pred_val,
            }
            print(
                f"(P={p}, Q={q}) -> OK | params={len(res_train.params)}"
                f" | BIC={res_train.bic:.2f}"
                f" | HQIC={res_train.hqic:.2f}"
                f" | AIC={res_train.aic:.2f}"
                f" | RMSE_val={row['RMSE_val']:.4f}"
            )
        except Exception as exc:
            row.update({
                "Status": f"FAIL: {type(exc).__name__}",
                "Num_Params": np.nan,
                "AIC": np.nan, "BIC": np.nan, "HQIC": np.nan,
                "RMSE_val": np.nan, "MAE_val": np.nan,
            })
            print(f"(P={p}, Q={q}) -> FAIL: {type(exc).__name__}: {exc}")

        sweep_rows.append(row)

    ardl_sweep_table = pd.DataFrame(sweep_rows)
    sweep_csv = sweep_dir / "sweep_results.csv"
    ardl_sweep_table.to_csv(sweep_csv, index=False)
    print("ARDL step 5: sweep saved to", sweep_csv)

    # Auto-select if SELECTED_PAIR not already set in context
    if "SELECTED_PAIR" not in context or context["SELECTED_PAIR"] is None:
        criterion = ardl_cfg.get("selection_criterion", "BIC")
        if criterion in ("RMSE_test", "MAE_test", "RMSE_trainval", "MAE_trainval"):
            raise ValueError(
                f"selection_criterion={criterion!r} requires Test/trainval "
                "metrics, which are never computed during candidate "
                "selection under the P0-3A protocol (Test must not "
                "influence model selection). Use BIC/AIC/HQIC/RMSE_val/"
                "MAE_val instead."
            )
        # Per-criterion information-criterion cap (used only for AIC/BIC/HQIC)
        threshold_map = {
            "AIC":  ardl_cfg.get("aic_thresholds"),
            "BIC":  ardl_cfg.get("bic_thresholds"),
            "HQIC": ardl_cfg.get("hqic_thresholds"),
        }
        criterion_threshold = threshold_map.get(criterion)
        rmse_val_threshold = ardl_cfg.get("rmse_val_thresholds")
        context["SELECTED_PAIR"] = select_best_model(
            ardl_sweep_table,
            criterion=criterion,
            criterion_threshold=criterion_threshold,
            rmse_val_threshold=rmse_val_threshold,
        )

    context.update({
        "trainval_df": trainval_df,
        "y_train": y_train,
        "X_train": X_train,
        "y_val": y_val,
        "X_val": X_val,
        "y_trainval": y_trainval,
        "X_trainval": X_trainval,
        "y_test": y_test,
        "X_test": X_test,
        "pq_pairs": pq_pairs,
        "ardl_results_by_pair": ardl_results_by_pair,
        "ardl_sweep_table": ardl_sweep_table,
        "sweep_csv": sweep_csv,
        "ardl_causal": causal,
        "ardl_hold_back": hold_back,
    })
    return context
