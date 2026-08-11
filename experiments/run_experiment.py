"""
run_experiment.py – Unified experiment entry point (canonical)
====================================================================
Runs the full VN-Index research pipeline: preprocess+PCA -> ARDL ->
LSTM -> Diebold-Mariano comparison. Replaces the former run_pipeline.py
as the canonical orchestration layer; run_pipeline.py is now a thin
backward-compatible shim that imports this module.

All outputs are collected under artifacts/Run_<YYYYMMDD_HHMMSS>/

Usage
-----
  # Standard single-CEV run (default):
  python experiments/run_experiment.py

  # Multi-CEV sweep across [0.85, 0.90, 0.95]:
  python experiments/run_experiment.py --multi-cev

  # Custom config:
  python experiments/run_experiment.py --config configs/config.yaml

  # Skip steps you've already run:
  python experiments/run_experiment.py --skip-preprocess

Flow
----
  [1] Preprocess + PCA          -> src/run_all.py
  [2] ARDL grid search          -> experiments/run_ardl_experiment.py
  [3] LSTM sweep                -> experiments/run_lstm_experiment.py
  [4] Diebold-Mariano test      -> src/evaluation/run_dm_test.py
  [5] Copy everything           -> artifacts/Run_<timestamp>/
  [6] Write run_manifest.json

  For --multi-cev, steps [1]-[5] are wrapped by src/run_all_multi_cev.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

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


# ── Output collection ────────────────────────────────────────────────────────

# Directories produced by each pipeline step
_PIPELINE_OUTPUTS = [
    # preprocess / PCA
    "data/processed",
    "models",
    "logs/figures",
    # ARDL
    "outputs/ardl_vnindex_pca_sweep",
    "outputs/ardl_vnindex_forecast",
    # LSTM
    "outputs/lstm_vnindex_sweep",
    # DM Test
    "outputs/model_comparison",
    # Multi-CEV
    "outputs/cev_comparison",
]


def _collect_outputs(run_dir: Path, multi_cev: bool, cev_thresholds: list[float]) -> None:
    """
    Copy pipeline outputs into run_dir, preserving sub-structure.
    Multi-CEV results are under outputs/cev_{x}/...
    """
    _hdr(f"Collecting outputs -> {run_dir}")

    dirs_to_copy = list(_PIPELINE_OUTPUTS)
    if multi_cev:
        for cev in cev_thresholds:
            dirs_to_copy.append(f"outputs/cev_{cev:.2f}")

    copied, skipped = 0, 0
    for rel in dirs_to_copy:
        src = PROJECT_ROOT / rel
        if not src.exists():
            print(f"  [SKIP] {rel}  (not found)")
            skipped += 1
            continue
        dst = run_dir / rel
        if dst.exists():
            shutil.rmtree(dst)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        print(f"  [OK]   {rel}")
        copied += 1

    print(f"\nCopied {copied} directories, skipped {skipped}.")


def _write_manifest(run_dir: Path, config: dict, args: argparse.Namespace,
                    timings: dict, failed_steps: list[str]) -> None:
    """Write run_manifest.json summarising the run."""
    manifest = {
        "run_dir":      str(run_dir),
        "timestamp":    run_dir.name,
        "config_path":  str(PROJECT_ROOT / args.config),
        "multi_cev":    args.multi_cev,
        "skip_preprocess": args.skip_preprocess,
        "pca": {
            "threshold":    config.get("pca", {}).get("explained_variance_threshold"),
            "cev_thresholds": config.get("pca", {}).get("cev_thresholds"),
        },
        "preprocess": {
            "train_ratio":      config.get("preprocess", {}).get("train_ratio"),
            "val_ratio":        config.get("preprocess", {}).get("val_ratio"),
            "use_overlap_val":  config.get("preprocess", {}).get("use_overlap_val"),
            "test_ratio":       config.get("preprocess", {}).get("test_ratio"),
        },
        "ardl": {
            "selection_criterion": config.get("ardl", {}).get("selection_criterion"),
            "selected_pair":       config.get("ardl", {}).get("selected_pair"),
            "use_ensemble":        config.get("ardl", {}).get("use_ensemble"),
        },
        "lstm": {
            "lookback_values":  config.get("lstm", {}).get("lookback_values"),
            "batch_size_values": config.get("lstm", {}).get("batch_size_values"),
            "learning_rate":    config.get("lstm", {}).get("learning_rate"),
            "epochs":           config.get("lstm", {}).get("epochs"),
            "use_batch_norm":   config.get("lstm", {}).get("use_batch_norm"),
        },
        "timings_seconds": timings,
        "failed_steps":    failed_steps,
        "status":          "FAILED" if failed_steps else "OK",
    }
    out = run_dir / "run_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n[COMP] Manifest: {out}")


# ── Single-CEV flow ───────────────────────────────────────────────────────────

def run_single_cev(args: argparse.Namespace, config: dict,
                   run_dir: Path) -> list[str]:
    """
    Run preprocess -> ARDL -> LSTM -> DM test sequentially.
    Returns list of failed step names.
    """
    failed: list[str] = []
    py = sys.executable
    cfg = str(PROJECT_ROOT / args.config)

    steps: list[tuple[str, list[str], Path | None]] = []

    if not args.skip_preprocess:
        steps.append((
            "Preprocess + PCA",
            [py, "src/run_all.py", "--config", cfg],
            PROJECT_ROOT,
        ))

    if not args.skip_ardl:
        steps.append((
            "ARDL",
            [py, "experiments/run_ardl_experiment.py", "--config", cfg],
            PROJECT_ROOT,
        ))

    if not args.skip_lstm:
        steps.append((
            "LSTM",
            [py, "experiments/run_lstm_experiment.py"],
            PROJECT_ROOT,
        ))

    if not args.skip_dm:
        steps.append((
            "DM Test",
            [py, "-m", "src.evaluation.run_dm_test"],
            PROJECT_ROOT,
        ))

    for label, cmd, cwd in steps:
        _hdr(f"STEP: {label}")
        rc = _run(cmd, cwd=cwd, label=label)
        if rc != 0:
            print(f"\n[FAIL] {label} failed (exit code {rc})")
            failed.append(label)
            if not args.continue_on_error:
                print("Stopping pipeline. Use --continue-on-error to skip failures.")
                break
        else:
            print(f"\n[PASS] {label} done")

    return failed


# ── Multi-CEV flow ────────────────────────────────────────────────────────────

def run_multi_cev(args: argparse.Namespace) -> list[str]:
    """Delegate to src/run_all_multi_cev.py and return failed steps."""
    _hdr("MULTI-CEV PIPELINE  ->  src/run_all_multi_cev.py")
    rc = _run(
        [sys.executable, "src/run_all_multi_cev.py", "--config", args.config],
        cwd=PROJECT_ROOT,
        label="Multi-CEV orchestrator",
    )
    return [] if rc == 0 else ["Multi-CEV orchestrator"]


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified VNIndex research experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/run_experiment.py                     # full single-CEV run
  python experiments/run_experiment.py --multi-cev         # CEV sweep [0.85, 0.90, 0.95]
  python experiments/run_experiment.py --skip-preprocess   # re-run models only
  python experiments/run_experiment.py --skip-ardl --skip-lstm  # DM test only
        """,
    )
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--multi-cev", action="store_true",
                   help="Run multi-CEV sweep (delegates to run_all_multi_cev.py)")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--skip-ardl", action="store_true")
    p.add_argument("--skip-lstm", action="store_true")
    p.add_argument("--skip-dm", action="store_true")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Continue to next step even if one fails")
    p.add_argument("--no-collect", action="store_true",
                   help="Skip copying outputs to artifacts/ (useful for dry runs)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"[FAIL] Config not found: {config_path}")
        sys.exit(1)

    config = _load_config(config_path)
    cev_thresholds = config.get("pca", {}).get("cev_thresholds", [0.85, 0.90, 0.95])

    # Auto-detect multi-CEV from config if not set via CLI
    if not args.multi_cev and config.get("pca", {}).get("use_multi_cev", False):
        print("[INFO] use_multi_cev=true detected in config -> switching to --multi-cev mode")
        args.multi_cev = True

    # Create timestamped run directory
    artifacts_base = PROJECT_ROOT / "artifacts"
    run_dir = _make_run_dir(artifacts_base)

    _hdr(f"VNIndex Experiment  |  {'Multi-CEV' if args.multi_cev else 'Single-CEV'}")
    print(f"  Config  : {config_path}")
    print(f"  Run dir : {run_dir}")
    print(f"  CEV     : {cev_thresholds if args.multi_cev else config.get('pca', {}).get('explained_variance_threshold')}")

    wall_start = time.time()
    timings: dict[str, float] = {}
    failed: list[str] = []

    # ── Run the pipeline ──────────────────────────────────────────
    if args.multi_cev:
        t0 = time.time()
        failed = run_multi_cev(args)
        timings["multi_cev_total"] = round(time.time() - t0, 1)
    else:
        t0 = time.time()
        failed = run_single_cev(args, config, run_dir)
        timings["pipeline_total"] = round(time.time() - t0, 1)

    # ── Collect outputs ───────────────────────────────────────────
    if not args.no_collect:
        _hdr("Collecting all outputs into run directory")
        _collect_outputs(run_dir, args.multi_cev, cev_thresholds)

    # ── Write manifest ────────────────────────────────────────────
    timings["wall_seconds"] = round(time.time() - wall_start, 1)
    _write_manifest(run_dir, config, args, timings, failed)

    # ── Final summary ─────────────────────────────────────────────
    _hdr("EXPERIMENT COMPLETE")
    status = "[COMP] OK" if not failed else f"[WARN]  FAILED steps: {', '.join(failed)}"
    print(f"  Status  : {status}")
    print(f"  Elapsed : {timings['wall_seconds']:.0f}s ({timings['wall_seconds']/60:.1f} min)")
    print(f"  Results : {run_dir}")
    print()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
