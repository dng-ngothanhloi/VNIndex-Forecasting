from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .common import paired_valid


# --- from step_04a_adf_stationarity_test.py ---
def run_adf_stationarity_test(context: dict) -> dict:
    """
    ADF (Augmented Dickey-Fuller) stationarity test
    Tests VNINDEX and all PCs for unit roots (I(0) vs I(1))
    """
    train_df = context["train_df"]
    val_df = context["val_df"]
    test_df = context["test_df"]
    target_col = context["target_col"]
    pc_cols = context["pc_cols"]

    # Concatenate all data for full-sample stationarity test
    full_df = pd.concat([train_df, val_df, test_df], axis=0).sort_index()

    print("\n" + "=" * 80)
    print("ARDL STEP 4A: STATIONARITY TEST (ADF Test)")
    print("=" * 80)

    adf_results = {}

    # Test VNINDEX (target variable)
    print(f"\nTesting {target_col}...")
    y_level = full_df[target_col].dropna().astype(float)

    adf_level = adfuller(y_level, autolag="AIC")
    adf_results[f"{target_col}_level"] = {
        "adf_stat": float(adf_level[0]),
        "pvalue": float(adf_level[1]),
        "usedlag": int(adf_level[2]),
        "nobs": int(adf_level[3]),
    }

    # Test first difference
    y_diff1 = y_level.diff().dropna()
    adf_diff1 = adfuller(y_diff1, autolag="AIC")
    adf_results[f"{target_col}_diff1"] = {
        "adf_stat": float(adf_diff1[0]),
        "pvalue": float(adf_diff1[1]),
        "usedlag": int(adf_diff1[2]),
        "nobs": int(adf_diff1[3]),
    }

    # Print results for target variable
    print(f"  ADF (mức gốc): {adf_level[0]:.6f}")
    print(f"  p-value (mức gốc): {adf_level[1]:.6f}")
    print(f"  ADF (sai phân bậc 1): {adf_diff1[0]:.6f}")
    print(f"  p-value (Δ): {adf_diff1[1]:.6f}")

    if adf_level[1] < 0.05:
        print(f"    → {target_col} is I(0) (stationary at level)")
    else:
        print(f"    → {target_col} is I(1) (NOT stationary at level)")

    if adf_diff1[1] < 0.05:
        print(f"    → {target_col} is I(0) after differencing (becomes stationary)")

    # Test each feature column
    # For high-dimensional representations (NoReduction, F>50), test only a
    # summary sample to avoid excessive runtime; report aggregate counts.
    _max_detailed_adf = 50
    _test_cols = pc_cols if len(pc_cols) <= _max_detailed_adf else pc_cols[:_max_detailed_adf]
    if len(pc_cols) > _max_detailed_adf:
        print(f"\nTesting first {_max_detailed_adf} of {len(pc_cols)} feature columns (NoReduction mode)...")
    else:
        print(f"\nTesting {len(pc_cols)} feature columns...")
    for pc_col in _test_cols:
        x_level = full_df[pc_col].dropna().astype(float)

        adf_level = adfuller(x_level, autolag="AIC")
        adf_results[f"{pc_col}_level"] = {
            "adf_stat": float(adf_level[0]),
            "pvalue": float(adf_level[1]),
            "usedlag": int(adf_level[2]),
            "nobs": int(adf_level[3]),
        }

        # Print results for each PC
        print(f"\n  {pc_col}:")
        print(f"    ADF (mức gốc): {adf_level[0]:.6f}")
        print(f"    p-value (mức gốc): {adf_level[1]:.6f}")

        # Check if stationary at level
        if adf_level[1] < 0.05:
            print(f"    → {pc_col} is I(0) (stationary at level)")
        else:
            # Test first difference
            x_diff1 = x_level.diff().dropna()
            adf_diff1 = adfuller(x_diff1, autolag="AIC")
            adf_results[f"{pc_col}_diff1"] = {
                "adf_stat": float(adf_diff1[0]),
                "pvalue": float(adf_diff1[1]),
                "usedlag": int(adf_diff1[2]),
                "nobs": int(adf_diff1[3]),
            }
            print(f"    ADF (sai phân bậc 1): {adf_diff1[0]:.6f}")
            print(f"    p-value (Δ): {adf_diff1[1]:.6f}")
            print(f"    → {pc_col} is I(1) at level, I(0) after differencing")

    # Summary statistics
    print("\n" + "-" * 80)
    print("SUMMARY:")
    print("-" * 80)

    i0_vars = sum(1 for k in adf_results if k.endswith("_level") and adf_results[k]["pvalue"] < 0.05)
    i1_vars = sum(1 for k in adf_results if k.endswith("_level") and adf_results[k]["pvalue"] >= 0.05)

    print(f"  I(0) variables (stationary at level):     {i0_vars}")
    print(f"  I(1) variables (need differencing):       {i1_vars}")
    print(f"  Total variables tested:                    {len(_test_cols) + 1}")

    # Check ARDL suitability
    ardl_suitable = True
    if target_col in [k.replace("_level", "") for k in adf_results if k.endswith("_level")]:
        # Check if VNINDEX is I(0) or I(1)
        vnindex_pvalue = adf_results[f"{target_col}_level"]["pvalue"]
        if vnindex_pvalue < 0.05:
            print(f"\n[WARN]  WARNING: {target_col} is already I(0) - ARDL may not be ideal")
        else:
            print(f"\n[PASS] {target_col} is I(1) - suitable for ARDL framework")

    # Check if we have at least some I(0) or I(1) mixed
    pc_orders = []
    for pc_col in _test_cols:
        pval = adf_results.get(f"{pc_col}_level", {}).get("pvalue")
        if pval is None:
            continue
        if pval < 0.05:
            pc_orders.append("I(0)")
        else:
            pc_orders.append("I(1)")

    if "I(1)" in pc_orders and "I(0)" in pc_orders:
        print("[PASS] Mixed I(0) and I(1) PCs - suitable for ARDL framework")
    elif all(order == "I(1)" for order in pc_orders):
        print("[PASS] All PCs are I(1) - suitable for ARDL framework (cointegration test)")
    elif all(order == "I(0)" for order in pc_orders):
        print("[WARN]  All PCs are I(0) - may not need ARDL, consider VAR or other models")

    print("=" * 80 + "\n")

    # Save results
    results_dir = context["PROJECT_ROOT"] / "outputs" / "ardl_vnindex_report"
    results_dir.mkdir(parents=True, exist_ok=True)
    adf_json_path = results_dir / "adf_results.json"

    with open(adf_json_path, "w", encoding="utf-8") as f:
        json.dump(adf_results, f, indent=2)

    print(f"ADF results saved to: {adf_json_path}")

    context.update({
        "adf_results": adf_results,
        "ardl_suitable": ardl_suitable,
    })

    return context
