from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

from pca_model import run_pca_pipeline, run_reduction_pipeline
from preprocess import preprocess_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full VNINDEX pipeline (preprocess + PCA)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_temp_config(config: dict, base_path: Path, cev: float) -> Path:
    """Write a temporary config with a single CEV threshold and return its path."""
    import copy
    cfg = copy.deepcopy(config)
    cfg["pca"]["explained_variance_threshold"] = cev
    cfg["pca"]["use_multi_cev"] = False          # avoid recursion
    cfg["pca"]["output_subdir"] = f"cev_{cev:.2f}"

    temp_path = base_path.parent / f"_temp_cev_{cev:.2f}.yaml"
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return temp_path


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / args.config
    config = _load_config(config_path)

    use_multi_cev   = config.get("pca", {}).get("use_multi_cev", False)
    cev_thresholds  = config.get("pca", {}).get("cev_thresholds", [0.75, 0.80, 0.85, 0.90, 0.95])

    # ── Step 1: Preprocess (always run once, shared across CEV levels) ────────
    print("[STEP 1/2] Running preprocessing pipeline...")
    preprocess_pipeline(project_root=project_root, config_path=config_path)

    # ── Step 2: Representation (PCA or NoReduction) ─────────────────────────
    reduction_method = config.get("reduction", {}).get("method", "pca")

    if reduction_method == "none":
        # NoReduction: output raw scaled features directly
        print("[STEP 2/2] Running NoReduction pipeline (identity representation)...")
        run_reduction_pipeline(project_root=project_root, config_path=config_path)
    elif use_multi_cev:
        print(f"[STEP 2/2] Running PCA pipeline for {len(cev_thresholds)} CEV thresholds: {cev_thresholds}")
        for cev in cev_thresholds:
            print(f"\n{'='*60}")
            print(f"  PCA  CEV = {cev:.2f}")
            print(f"{'='*60}")
            temp_cfg = _save_temp_config(config, config_path, cev)
            try:
                run_pca_pipeline(project_root=project_root, config_path=temp_cfg)
            finally:
                temp_cfg.unlink(missing_ok=True)
        print(f"\n[INFO] Multi-CEV PCA complete. "
              f"Active pca/ dir reflects CEV={cev_thresholds[-1]:.2f}. "
              f"Per-CEV archives: data/processed/pca_cev_X.XX/")
    else:
        print("[STEP 2/2] Running PCA pipeline...")
        run_pca_pipeline(project_root=project_root, config_path=config_path)

    print("\n[DONE] Full pipeline completed successfully.")


if __name__ == "__main__":
    main()
