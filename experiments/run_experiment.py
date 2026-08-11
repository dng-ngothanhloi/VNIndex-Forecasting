"""
run_experiment.py – Unified experiment entry point (canonical)
====================================================================
Runs the full VN-Index research pipeline: preprocess+representation ->
ARDL -> LSTM -> Diebold-Mariano comparison.

Artifact structure (unified for single and sweep runs):
  artifacts/Run_<YYYYMMDD_HHMMSS>/
  ├── sweep_manifest.json           (sweep metadata)
  ├── no_dr/                        (child: NoReduction)
  │   ├── run_manifest.json
  │   ├── results/run_summary.json
  │   ├── config/effective_config.yaml
  │   ├── data/processed/...
  │   ├── models/...
  │   ├── outputs/...
  │   └── logs/...
  ├── pca_cev_0.75/                 (child: PCA CEV=0.75)
  │   └── ...
  └── comparison/                   (aggregated results)
      └── representation_comparison.csv

Usage
-----
  # Single run (one label inside one Run_*):
  python experiments/run_experiment.py --reduction pca --cev 0.75

  # Full sweep (NoReduction + all CEV levels, one Run_*):
  python experiments/run_experiment.py --full-sweep

  # Compare results from a sweep:
  python -m src.evaluation.compare_representations --run-dir artifacts/Run_<ts>
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hdr(text: str, width: int = 72) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _run(cmd: list[str], cwd: Path | None = None, label: str = "") -> int:
    """Run subprocess, stream output, return exit code."""
    if label:
        print(f"\n[RUN] {label}")
    print(f"      {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd or PROJECT_ROOT))
    return result.returncode


def _make_run_dir(base: Path) -> Path:
    """Create artifacts/Run_<YYYYMMDD_HHMMSS>/ and return its path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"Run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _label_for(method: str, cev: float | None) -> str:
    """Deterministic label for a representation config."""
    if method == "none":
        return "no_dr"
    if cev is not None:
        return f"pca_cev_{cev:.2f}"
    return "pca_unknown"


# ── Mutable output dirs produced by pipeline steps ───────────────────────────
_MUTABLE_OUTPUT_DIRS = [
    "data/processed",
    "models",
    "logs/figures",
    "outputs/ardl_vnindex_pca_sweep",
    "outputs/ardl_vnindex_forecast",
    "outputs/ardl_vnindex_report",
    "outputs/lstm_vnindex_sweep",
    "outputs/model_comparison",
]


def _clean_mutable_outputs() -> None:
    """Remove known mutable outputs before a child run to prevent stale data."""
    for rel in _MUTABLE_OUTPUT_DIRS:
        p = PROJECT_ROOT / rel
        if p.exists():
            shutil.rmtree(p)


def _snapshot_child(child_dir: Path) -> None:
    """Copy mutable outputs into the child snapshot directory."""
    for rel in _MUTABLE_OUTPUT_DIRS:
        src = PROJECT_ROOT / rel
        if not src.exists():
            continue
        dst = child_dir / rel
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def _read_pca_metrics() -> dict:
    """Read pca_metrics.csv from the live data/processed/pca/ after a run."""
    metrics_path = PROJECT_ROOT / "data" / "processed" / "pca" / "pca_metrics.csv"
    if not metrics_path.exists():
        return {}
    try:
        df = pd.read_csv(metrics_path, index_col=0)
        return df["value"].to_dict()
    except Exception:
        return {}


def _read_ardl_summary() -> dict:
    """Read ARDL test metrics from forecast artifacts."""
    forecast_path = PROJECT_ROOT / "outputs" / "ardl_vnindex_forecast" / "chapter4_ardl_forecast.csv"
    info: dict = {}
    if forecast_path.exists():
        try:
            df = pd.read_csv(forecast_path)
            if "Actual_VNINDEX" in df.columns and "Predicted_VNINDEX" in df.columns:
                from sklearn.metrics import mean_absolute_error, r2_score
                actual = df["Actual_VNINDEX"].values
                pred = df["Predicted_VNINDEX"].values
                res = actual - pred
                info["ARDL_RMSE"] = float(np.sqrt(np.mean(res ** 2)))
                info["ARDL_MAE"] = float(mean_absolute_error(actual, pred))
                info["ARDL_MAPE"] = float(np.mean(np.abs(res / (actual + 1e-8))) * 100)
                info["ARDL_R2"] = float(r2_score(actual, pred))
                info["ARDL_N"] = len(df)
        except Exception:
            pass
    # Selected pair from sweep
    sweep_path = PROJECT_ROOT / "outputs" / "ardl_vnindex_pca_sweep" / "sweep_results.csv"
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


def _read_lstm_summary() -> dict:
    """Read LSTM test metrics from sweep artifacts."""
    lstm_dir = PROJECT_ROOT / "outputs" / "lstm_vnindex_sweep"
    info: dict = {}
    summary_path = lstm_dir / "sweep_summary.csv"
    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path)
            best_row = summary.loc[summary["Val_RMSE"].idxmin()]
            info["LSTM_LB"] = int(best_row["Lookback"])
            info["LSTM_BS"] = int(best_row["Batch_size"])
            info["LSTM_best_epoch"] = int(best_row["Best_Epoch"])
        except Exception:
            pass
    for pred_file in sorted(lstm_dir.glob("predictions_lookback_*.csv")):
        try:
            df = pd.read_csv(pred_file)
            if "Actual_VNINDEX" in df.columns and "Predicted_VNINDEX" in df.columns:
                from sklearn.metrics import mean_absolute_error, r2_score
                actual = df["Actual_VNINDEX"].values
                pred = df["Predicted_VNINDEX"].values
                res = actual - pred
                info["LSTM_RMSE"] = float(np.sqrt(np.mean(res ** 2)))
                info["LSTM_MAE"] = float(mean_absolute_error(actual, pred))
                info["LSTM_MAPE"] = float(np.mean(np.abs(res / (actual + 1e-8))) * 100)
                info["LSTM_R2"] = float(r2_score(actual, pred))
                info["LSTM_N"] = len(df)
                break
        except Exception:
            continue
    return info


def _read_dm_summary() -> dict:
    """Read DM test p-values."""
    dm_path = PROJECT_ROOT / "outputs" / "model_comparison" / "dm_test_results.csv"
    if not dm_path.exists():
        return {}
    try:
        dm = pd.read_csv(dm_path)
        info: dict = {}
        for _, row in dm.iterrows():
            loss = row.get("Loss_Type", "")
            if loss == "MSE":
                info["DM_MSE_p"] = row.get("p_value")
            elif loss == "MAE":
                info["DM_MAE_p"] = row.get("p_value")
        return info
    except Exception:
        return {}


# ── Child execution ──────────────────────────────────────────────────────────

def _run_child(label: str, method: str, cev: float | None,
               parent_run_dir: Path, config: dict, config_path: Path) -> str:
    """Execute one child run (preprocess+repr → ARDL → LSTM → DM).
    
    Returns status: "OK" or "FAILED".
    """
    child_dir = parent_run_dir / label
    child_dir.mkdir(parents=True, exist_ok=True)

    # Write effective config for this child
    child_config = copy.deepcopy(config)
    child_config.setdefault("reduction", {})["method"] = method
    if cev is not None:
        child_config.setdefault("pca", {})["explained_variance_threshold"] = cev

    config_dir = child_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    effective_cfg_path = config_dir / "effective_config.yaml"
    with open(effective_cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(child_config, f, default_flow_style=False, allow_unicode=True)

    # Write temp config for subprocesses
    temp_cfg_path = config_path.parent / f"_temp_{label}_{int(time.time())}.yaml"
    with open(temp_cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(child_config, f, default_flow_style=False, allow_unicode=True)

    py = sys.executable
    failed_steps: list[str] = []
    t0 = time.time()

    try:
        # Clean mutable outputs to prevent stale data from previous child
        _clean_mutable_outputs()

        # Step 1: Preprocess + Representation (ALWAYS run — each child needs its own repr)
        _hdr(f"[{label}] Preprocess + Representation")
        rc = _run([py, "src/run_all.py", "--config", str(temp_cfg_path)],
                  cwd=PROJECT_ROOT, label=f"{label}: Preprocess+Repr")
        if rc != 0:
            failed_steps.append("Preprocess+Repr")

        # Step 2: ARDL
        if not failed_steps:
            _hdr(f"[{label}] ARDL")
            rc = _run([py, "experiments/run_ardl_experiment.py", "--config", str(temp_cfg_path)],
                      cwd=PROJECT_ROOT, label=f"{label}: ARDL")
            if rc != 0:
                failed_steps.append("ARDL")

        # Step 3: LSTM
        if not failed_steps:
            _hdr(f"[{label}] LSTM")
            rc = _run([py, "experiments/run_lstm_experiment.py"],
                      cwd=PROJECT_ROOT, label=f"{label}: LSTM")
            if rc != 0:
                failed_steps.append("LSTM")

        # Step 4: DM Test
        if not failed_steps:
            _hdr(f"[{label}] DM Test")
            rc = _run([py, "-m", "src.evaluation.run_dm_test"],
                      cwd=PROJECT_ROOT, label=f"{label}: DM")
            if rc != 0:
                failed_steps.append("DM")

    finally:
        temp_cfg_path.unlink(missing_ok=True)

    elapsed = time.time() - t0
    status = "FAILED" if failed_steps else "OK"

    # Snapshot mutable outputs into child dir
    _snapshot_child(child_dir)

    # Read actual metrics from live outputs (before next child cleans them)
    pca_metrics = _read_pca_metrics()
    ardl_summary = _read_ardl_summary()
    lstm_summary = _read_lstm_summary()
    dm_summary = _read_dm_summary()

    # Write child run_summary.json
    results_dir = child_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "label": label,
        "status": status,
        "failed_steps": failed_steps,
        "elapsed_seconds": round(elapsed, 1),
        "representation": {
            "method": method,
            "cev_requested": cev,
            "cev_achieved": pca_metrics.get("cev_achieved"),
            "p_original": pca_metrics.get("input_features"),
            "k_actual": pca_metrics.get("k_optimal"),
            "dim_reduction_pct": pca_metrics.get("dim_reduction_pct"),
        },
        "ardl": ardl_summary,
        "lstm": lstm_summary,
        "dm": dm_summary,
    }
    with open(results_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, default=str)

    # Write child run_manifest.json
    child_manifest = {
        "parent_run_id": parent_run_dir.name,
        "label": label,
        "status": status,
        "effective_config_path": str(effective_cfg_path.relative_to(parent_run_dir)),
        "summary_path": str((results_dir / "run_summary.json").relative_to(parent_run_dir)),
        "representation": run_summary["representation"],
        "elapsed_seconds": round(elapsed, 1),
        "failed_steps": failed_steps,
    }
    with open(child_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(child_manifest, f, indent=2, default=str)

    print(f"\n  [{status}] {label}  ({elapsed:.0f}s)")
    return status


# ── Full Sweep ───────────────────────────────────────────────────────────────

def _run_full_sweep(args: argparse.Namespace, config: dict, config_path: Path) -> None:
    """Run NoReduction + all PCA CEV levels under ONE parent Run_* directory.
    
    Each (representation, cev) combination is a child label subdirectory
    with its own complete, self-contained artifacts snapshot.
    """
    cev_thresholds = config.get("pca", {}).get("cev_thresholds", [0.75, 0.80, 0.85, 0.90, 0.95])

    # Build deterministic label sequence
    sweep_configs: list[tuple[str, float | None]] = [("none", None)]
    sweep_configs += [("pca", cev) for cev in cev_thresholds]
    planned_labels = [_label_for(m, c) for m, c in sweep_configs]

    # Create ONE parent directory
    artifacts_base = PROJECT_ROOT / "artifacts"
    parent_run_dir = _make_run_dir(artifacts_base)

    _hdr(f"FULL SWEEP: {len(sweep_configs)} representations")
    print(f"  Parent : {parent_run_dir}")
    print(f"  Labels : {', '.join(planned_labels)}")
    print()

    wall_start = time.time()
    completed_labels: list[str] = []
    failed_labels: list[str] = []

    for i, (method, cev) in enumerate(sweep_configs, 1):
        label = _label_for(method, cev)
        _hdr(f"CHILD {i}/{len(sweep_configs)}: {label}")

        status = _run_child(label, method, cev, parent_run_dir, config, config_path)
        if status == "OK":
            completed_labels.append(label)
        else:
            failed_labels.append(label)

    wall_elapsed = time.time() - wall_start

    # Write parent sweep_manifest.json
    sweep_manifest = {
        "schema_version": 2,
        "run_id": parent_run_dir.name,
        "experiment_type": "representation_sweep",
        "started_at": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat(),
        "wall_seconds": round(wall_elapsed, 1),
        "planned_labels": planned_labels,
        "completed_labels": completed_labels,
        "failed_labels": failed_labels,
        "common_protocol": {
            "train_ratio": config.get("preprocess", {}).get("train_ratio"),
            "val_ratio": config.get("preprocess", {}).get("val_ratio"),
            "test_ratio": config.get("preprocess", {}).get("test_ratio"),
            "forecast_horizon": "T+1",
        },
        "status": "OK" if not failed_labels else ("PARTIAL" if completed_labels else "FAILED"),
    }
    with open(parent_run_dir / "sweep_manifest.json", "w", encoding="utf-8") as f:
        json.dump(sweep_manifest, f, indent=2, default=str)

    _hdr(f"FULL SWEEP COMPLETE  |  {wall_elapsed/60:.1f} min")
    print(f"  Parent dir : {parent_run_dir}")
    print(f"  Completed  : {len(completed_labels)}/{len(planned_labels)}")
    if failed_labels:
        print(f"  Failed     : {', '.join(failed_labels)}")
    print(f"\n  Compare: python -m src.evaluation.compare_representations "
          f"--run-dir {parent_run_dir}")


# ── Single run (produces one child label inside a parent) ────────────────────

def _run_single(args: argparse.Namespace, config: dict, config_path: Path) -> None:
    """Run a single representation config. Produces one child label inside
    a fresh parent Run_* directory (same schema as full-sweep)."""
    method = config.get("reduction", {}).get("method", "pca")
    cev = config.get("pca", {}).get("explained_variance_threshold") if method == "pca" else None
    label = _label_for(method, cev)

    artifacts_base = PROJECT_ROOT / "artifacts"
    parent_run_dir = _make_run_dir(artifacts_base)

    _hdr(f"SINGLE RUN: {label}")
    print(f"  Parent : {parent_run_dir}")
    print()

    wall_start = time.time()
    status = _run_child(label, method, cev, parent_run_dir, config, config_path)
    wall_elapsed = time.time() - wall_start

    # Write parent sweep_manifest (single-child sweep)
    sweep_manifest = {
        "schema_version": 2,
        "run_id": parent_run_dir.name,
        "experiment_type": "single_run",
        "started_at": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat(),
        "wall_seconds": round(wall_elapsed, 1),
        "planned_labels": [label],
        "completed_labels": [label] if status == "OK" else [],
        "failed_labels": [label] if status != "OK" else [],
        "common_protocol": {
            "train_ratio": config.get("preprocess", {}).get("train_ratio"),
            "val_ratio": config.get("preprocess", {}).get("val_ratio"),
            "test_ratio": config.get("preprocess", {}).get("test_ratio"),
            "forecast_horizon": "T+1",
        },
        "status": status,
    }
    with open(parent_run_dir / "sweep_manifest.json", "w", encoding="utf-8") as f:
        json.dump(sweep_manifest, f, indent=2, default=str)

    _hdr("EXPERIMENT COMPLETE")
    print(f"  Status  : {status}")
    print(f"  Elapsed : {wall_elapsed:.0f}s ({wall_elapsed/60:.1f} min)")
    print(f"  Results : {parent_run_dir / label}")
    print()
    if status != "OK":
        sys.exit(1)


# ── Legacy flows (backward compat) ──────────────────────────────────────────

def _run_legacy_multi_cev(args: argparse.Namespace) -> list[str]:
    """Delegate to src/run_all_multi_cev.py (legacy)."""
    _hdr("LEGACY MULTI-CEV  ->  src/run_all_multi_cev.py")
    rc = _run(
        [sys.executable, "src/run_all_multi_cev.py", "--config", args.config],
        cwd=PROJECT_ROOT, label="Multi-CEV orchestrator",
    )
    return [] if rc == 0 else ["Multi-CEV orchestrator"]


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified VNIndex research experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/run_experiment.py --reduction pca --cev 0.75   # single PCA run
  python experiments/run_experiment.py --reduction none             # single NoReduction run
  python experiments/run_experiment.py --full-sweep                 # no_dr + all CEV (ONE Run_*)
  python experiments/run_experiment.py --multi-cev                  # legacy multi-CEV
  python -m src.evaluation.compare_representations --run-dir artifacts/Run_<ts>
        """,
    )
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--full-sweep", action="store_true",
                   help="Run NoReduction + all PCA CEV levels under ONE parent Run_*")
    p.add_argument("--multi-cev", action="store_true",
                   help="Legacy multi-CEV (delegates to run_all_multi_cev.py)")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--skip-ardl", action="store_true")
    p.add_argument("--skip-lstm", action="store_true")
    p.add_argument("--skip-dm", action="store_true")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Continue to next step even if one fails")
    p.add_argument("--no-collect", action="store_true",
                   help="Skip copying outputs to artifacts/ (unused in new schema)")
    p.add_argument("--reduction", type=str, default=None, choices=["pca", "none"],
                   help="Override reduction.method in config (pca or none)")
    p.add_argument("--cev", type=float, default=None,
                   help="Override pca.explained_variance_threshold (e.g. 0.75, 0.85)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"[FAIL] Config not found: {config_path}")
        sys.exit(1)

    config = _load_config(config_path)

    # Apply CLI overrides
    if args.reduction is not None:
        config.setdefault("reduction", {})["method"] = args.reduction
    if args.cev is not None:
        config.setdefault("pca", {})["explained_variance_threshold"] = args.cev

    # Dispatch
    if args.full_sweep:
        _run_full_sweep(args, config, config_path)
    elif args.multi_cev:
        _run_legacy_multi_cev(args)
    else:
        _run_single(args, config, config_path)


if __name__ == "__main__":
    main()
