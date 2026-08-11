"""
compare_representations.py – Representation comparison from a grouped sweep
============================================================================
Reads child labels from ONE specific artifacts/Run_* directory and produces
a comparison table. Does NOT scan all Run_* globally — requires --run-dir.

Usage:
    python -m src.evaluation.compare_representations --run-dir artifacts/Run_20260811_120500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_from_run_dir(run_dir: Path) -> pd.DataFrame:
    """Read child label results from a single parent Run_* directory.
    
    Only inspects direct child directories that contain run_manifest.json.
    Does NOT scan other Run_* dirs. Requires each child to have
    results/run_summary.json for metrics.
    """
    sweep_manifest = _load_json(run_dir / "sweep_manifest.json")
    if sweep_manifest is None:
        print(f"[WARN] No sweep_manifest.json in {run_dir}")
        return pd.DataFrame()

    parent_run_id = sweep_manifest.get("run_id", run_dir.name)
    planned = sweep_manifest.get("planned_labels", [])

    rows = []
    found_labels = set()

    for child_dir in sorted(run_dir.iterdir()):
        if not child_dir.is_dir():
            continue
        child_manifest = _load_json(child_dir / "run_manifest.json")
        if child_manifest is None:
            continue

        label = child_manifest.get("label", child_dir.name)

        # Duplicate label check
        if label in found_labels:
            print(f"[FAIL] Duplicate label '{label}' in {run_dir} — refusing to aggregate")
            sys.exit(1)
        found_labels.add(label)

        # Only include completed children
        if child_manifest.get("status") != "OK":
            print(f"  [SKIP] {label}: status={child_manifest.get('status')}")
            continue

        # Validate parent_run_id matches
        if child_manifest.get("parent_run_id") != parent_run_id:
            print(f"  [SKIP] {label}: parent_run_id mismatch "
                  f"(expected {parent_run_id}, got {child_manifest.get('parent_run_id')})")
            continue

        # Read run_summary
        summary = _load_json(child_dir / "results" / "run_summary.json")
        if summary is None:
            print(f"  [SKIP] {label}: missing results/run_summary.json")
            continue

        repr_info = summary.get("representation", {})
        ardl_info = summary.get("ardl", {})
        lstm_info = summary.get("lstm", {})
        dm_info = summary.get("dm", {})

        # PCA consistency check: label/manifest/summary CEV must agree
        manifest_cev = child_manifest.get("representation", {}).get("cev_requested")
        summary_cev = repr_info.get("cev_requested")
        if repr_info.get("method") == "pca":
            expected_label = f"pca_cev_{manifest_cev:.2f}" if manifest_cev is not None else None
            if expected_label and expected_label != label:
                print(f"  [INVALID] {label}: label/manifest CEV mismatch (expected {expected_label})")
                continue
            if manifest_cev is not None and summary_cev is not None:
                if abs(float(manifest_cev) - float(summary_cev)) > 0.001:
                    print(f"  [INVALID] {label}: manifest/summary CEV mismatch "
                          f"({manifest_cev} vs {summary_cev})")
                    continue

        # NoReduction cannot contain PCA k semantics
        if repr_info.get("method") == "none":
            k = repr_info.get("k_actual")
            p = repr_info.get("p_original")
            if k is not None and p is not None and k != p:
                print(f"  [INVALID] {label}: NoReduction but k_actual ({k}) != p_original ({p})")
                continue

        row = {
            "parent_run_id": parent_run_id,
            "label": label,
            "representation": repr_info.get("method"),
            "cev_requested": repr_info.get("cev_requested"),
            "cev_achieved": repr_info.get("cev_achieved"),
            "p_original": repr_info.get("p_original"),
            "k": repr_info.get("k_actual"),
            "dim_reduction_pct": repr_info.get("dim_reduction_pct"),
        }
        # ARDL
        row["ARDL_P"] = ardl_info.get("ARDL_P")
        row["ARDL_Q"] = ardl_info.get("ARDL_Q")
        row["ARDL_RMSE"] = ardl_info.get("ARDL_RMSE")
        row["ARDL_MAE"] = ardl_info.get("ARDL_MAE")
        row["ARDL_R2"] = ardl_info.get("ARDL_R2")
        row["ARDL_N"] = ardl_info.get("ARDL_N")
        # LSTM
        row["LSTM_LB"] = lstm_info.get("LSTM_LB")
        row["LSTM_BS"] = lstm_info.get("LSTM_BS")
        row["LSTM_best_epoch"] = lstm_info.get("LSTM_best_epoch")
        row["LSTM_RMSE"] = lstm_info.get("LSTM_RMSE")
        row["LSTM_MAE"] = lstm_info.get("LSTM_MAE")
        row["LSTM_R2"] = lstm_info.get("LSTM_R2")
        row["LSTM_N"] = lstm_info.get("LSTM_N")
        # DM
        row["DM_MSE_p"] = dm_info.get("DM_MSE_p")
        row["DM_MAE_p"] = dm_info.get("DM_MAE_p")

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort: no_dr first, then by cev_requested ascending
    df["_sort_method"] = df["representation"].map({"none": 0, "pca": 1}).fillna(2)
    df["_sort_cev"] = pd.to_numeric(df["cev_requested"], errors="coerce").fillna(-1)
    df = df.sort_values(["_sort_method", "_sort_cev"]).drop(columns=["_sort_method", "_sort_cev"])
    df = df.reset_index(drop=True)

    return df


def print_comparison_table(df: pd.DataFrame, run_dir: Path) -> None:
    """Print a formatted comparison table."""
    if df.empty:
        print("[INFO] No valid completed children found to compare.")
        return

    print("\n" + "=" * 120)
    print(f"  REPRESENTATION COMPARISON TABLE  |  {run_dir.name}")
    print("=" * 120)

    # Primary metrics
    primary_cols = [
        "label", "representation", "cev_requested", "k", "dim_reduction_pct",
        "ARDL_RMSE", "LSTM_RMSE", "ARDL_MAE", "LSTM_MAE",
        "ARDL_R2", "LSTM_R2", "DM_MSE_p", "DM_MAE_p",
    ]
    primary_cols = [c for c in primary_cols if c in df.columns]
    print(df[primary_cols].to_string(index=False))
    print()

    # Model selection detail
    detail_cols = ["label", "ARDL_P", "ARDL_Q", "LSTM_LB", "LSTM_BS",
                   "LSTM_best_epoch", "ARDL_N", "LSTM_N"]
    detail_cols = [c for c in detail_cols if c in df.columns]
    if len(detail_cols) > 1:
        print("  MODEL SELECTION DETAIL:")
        print(df[detail_cols].to_string(index=False))

    print("=" * 120)
    print(f"  Source: {run_dir}")
    print(f"  Children: {len(df)} valid / {len(list(run_dir.iterdir()))} total subdirs")
    print()


def save_comparison(df: pd.DataFrame, run_dir: Path) -> None:
    """Save comparison results under <run_dir>/comparison/."""
    comp_dir = run_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    csv_path = comp_dir / "representation_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")

    json_path = comp_dir / "representation_comparison.json"
    df.to_json(json_path, orient="records", indent=2)
    print(f"[SAVED] {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare representation results from a grouped sweep run"
    )
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to a specific artifacts/Run_* directory (REQUIRED)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save comparison CSV/JSON (print only)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        print(f"[FAIL] Run directory not found: {run_dir}")
        sys.exit(1)

    df = collect_from_run_dir(run_dir)
    print_comparison_table(df, run_dir)

    if not args.no_save and not df.empty:
        save_comparison(df, run_dir)


if __name__ == "__main__":
    main()
