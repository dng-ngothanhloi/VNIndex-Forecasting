"""
run_all_multi_cev.py – Multi-CEV pipeline orchestrator
=======================================================
Runs the full pipeline (preprocess once → PCA × 3 thresholds →
ARDL + LSTM per threshold → DM test per threshold) and saves results
under outputs/cev_{threshold}/.

Usage:
    cd /path/to/VNIndexPredictor
    source .venv/bin/activate
    python src/run_all_multi_cev.py
    python src/run_all_multi_cev.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR      = PROJECT_ROOT / "src"

# Ensure src is on path so we can import pca_model, preprocess directly
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _hdr(text: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {text}")
    print("=" * 72)


def _run_subprocess(cmd: list[str], cwd: Path) -> int:
    """Run a command as a subprocess, streaming stdout/stderr live."""
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_temp_config(config: dict, config_path: Path, cev: float) -> Path:
    """Write a per-CEV config file and return its path."""
    cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in config.items()}
    cfg["pca"] = {k: v for k, v in config["pca"].items()}
    cfg["pca"]["explained_variance_threshold"] = cev
    cfg["pca"]["output_subdir"] = f"cev_{cev:.2f}"

    temp_path = config_path.parent / f"_temp_cev_{cev:.2f}.yaml"
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return temp_path


def redirect_pca_outputs(project_root: Path, cev: float) -> None:
    """
    Copy PCA outputs (train/val/test_pca.csv) from data/processed/pca/
    to outputs/cev_{cev:.2f}/pca/ so ARDL and LSTM can pick them up
    from the standard data/processed/pca/ path during this CEV run.
    """
    src_pca = project_root / "data" / "processed" / "pca"
    dst_pca = project_root / "outputs" / f"cev_{cev:.2f}" / "pca"
    dst_pca.mkdir(parents=True, exist_ok=True)
    for f in src_pca.glob("*.csv"):
        shutil.copy2(f, dst_pca / f.name)
    for f in src_pca.glob("*.pkl"):
        shutil.copy2(f, dst_pca / f.name)
    print(f"[CEV] PCA outputs archived → {dst_pca}")


def redirect_model_outputs(project_root: Path, cev: float) -> None:
    """
    Move ARDL + LSTM outputs to outputs/cev_{cev:.2f}/ after each run
    so successive CEV runs don't overwrite each other.
    """
    cev_dir = project_root / "outputs" / f"cev_{cev:.2f}"

    for src_subdir in ["ardl_vnindex_pca_sweep", "ardl_vnindex_forecast",
                        "lstm_vnindex_sweep", "model_comparison"]:
        src = project_root / "outputs" / src_subdir
        if src.exists():
            dst = cev_dir / src_subdir
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"[CEV] {src_subdir} archived → {dst}")


def run_multi_cev(config_path: Path) -> None:
    config = load_config(config_path)
    cev_thresholds = config.get("pca", {}).get("cev_thresholds", [0.85, 0.90, 0.95])

    _hdr(f"MULTI-CEV PIPELINE  |  thresholds: {cev_thresholds}")
    total_start = time.time()

    # ── Step 0: Preprocess (once, shared across all CEV levels) ──
    _hdr("STEP 0 / Preprocessing  (shared)")
    rc = _run_subprocess(
        [sys.executable, "src/run_all.py", "--config", str(config_path)],
        cwd=PROJECT_ROOT,
    )
    if rc != 0:
        print("[FAIL] Preprocessing failed — aborting multi-CEV run.")
        sys.exit(1)

    # ── Loop over CEV thresholds ──────────────────────────────────
    for i, cev in enumerate(cev_thresholds, 1):
        _hdr(f"CEV {i}/{len(cev_thresholds)}  |  threshold = {cev:.2f}")
        cev_start = time.time()

        temp_cfg = save_temp_config(config, config_path, cev)
        print(f"[CEV] Temp config: {temp_cfg.name}")

        # PCA with this CEV threshold
        print(f"\n--- PCA (CEV={cev:.2f}) ---")
        rc = _run_subprocess(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0,'{SRC_DIR}');"
             f"from pca_model import run_pca_pipeline; from pathlib import Path;"
             f"run_pca_pipeline(Path('{PROJECT_ROOT}'), Path('{temp_cfg}'))"],
            cwd=PROJECT_ROOT,
        )
        if rc != 0:
            print(f"[FAIL] PCA failed for CEV={cev:.2f} — skipping")
            temp_cfg.unlink(missing_ok=True)
            continue

        # Archive PCA outputs before ARDL/LSTM overwrite them
        redirect_pca_outputs(PROJECT_ROOT, cev)

        # ARDL
        print(f"\n--- ARDL (CEV={cev:.2f}) ---")
        rc = _run_subprocess(
            [sys.executable, "experiments/run_ardl_experiment.py", "--config", str(temp_cfg)],
            cwd=PROJECT_ROOT,
        )
        if rc != 0:
            print(f"[FAIL] ARDL failed for CEV={cev:.2f}")

        # LSTM
        print(f"\n--- LSTM (CEV={cev:.2f}) ---")
        rc = _run_subprocess(
            [sys.executable, "experiments/run_lstm_experiment.py"],
            cwd=PROJECT_ROOT,
        )
        if rc != 0:
            print(f"[FAIL] LSTM failed for CEV={cev:.2f}")

        # DM Test
        print(f"\n--- DM Test (CEV={cev:.2f}) ---")
        rc = _run_subprocess(
            [sys.executable, "-m", "src.evaluation.run_dm_test"],
            cwd=PROJECT_ROOT,
        )
        if rc != 0:
            print(f"[FAIL] DM test failed for CEV={cev:.2f}")

        # Archive all outputs to cev_{threshold}/
        redirect_model_outputs(PROJECT_ROOT, cev)

        # Cleanup temp config
        temp_cfg.unlink(missing_ok=True)

        elapsed = time.time() - cev_start
        print(f"\n[COMP] CEV={cev:.2f} completed in {elapsed:.0f}s")

    # ── Final: CEV comparison ─────────────────────────────────────
    compare_script = PROJECT_ROOT / "src" / "evaluation" / "compare_representations.py"
    if compare_script.exists():
        _hdr("FINAL: Representation comparison")
        _run_subprocess(
            [sys.executable, "-m", "src.evaluation.compare_representations"],
            cwd=PROJECT_ROOT,
        )
    else:
        print("\n[INFO] compare_representations.py not found — skipping comparison")

    total_elapsed = time.time() - total_start
    _hdr(f"MULTI-CEV PIPELINE COMPLETE  |  {total_elapsed/60:.1f} min")
    print(f"Results in: {PROJECT_ROOT / 'outputs'}/cev_*")
    print("Run comparison: python -m src.evaluation.compare_representations")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-CEV pipeline orchestrator")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config YAML (default: configs/config.yaml)")
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    if not cfg_path.exists():
        print(f"[FAIL] Config not found: {cfg_path}")
        sys.exit(1)

    run_multi_cev(cfg_path)


if __name__ == "__main__":
    main()
