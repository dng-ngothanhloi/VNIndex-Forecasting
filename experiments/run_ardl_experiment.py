"""
run_ardl_experiment.py – ARDL experiment entry point (canonical)
====================================================================
Runs the scientifically-corrected ARDL(P,Q)+PCA sweep-and-forecast pipeline
(src/forecasting/ardl/*) end to end. This is the canonical replacement for
the former ardl/run_all_ardl.py.

Usage:
    python experiments/run_ardl_experiment.py
    python experiments/run_ardl_experiment.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.ardl.setup import run_setup as step_01_setup
from src.forecasting.ardl.setup import run_find_project_root as step_02_find_project_root
from src.forecasting.ardl.data import run_load_data as step_03_load_data
from src.forecasting.ardl.data import run_validate_pca as step_04_validate_pca
from src.forecasting.ardl.diagnostics import run_adf_stationarity_test as step_04a_adf_stationarity_test
from src.forecasting.ardl.sweep import run_sweep_ardl as step_05_sweep_ardl
from src.forecasting.ardl.forecast import run_select_and_forecast as step_06_select_and_forecast
from src.forecasting.ardl.export import run_export_pkl as step_07_export_pkl
from src.forecasting.ardl.reporting import run_summary as step_08_summary
from src.forecasting.ardl.reporting import run_plot as step_09_plot
from src.forecasting.ardl.reporting import run_ardl_80obs as step_10_ardl_80obs
from src.forecasting.ardl.reporting import run_summary_table as step_11_summary_table


def _load_selected_pair(config_path: Path) -> tuple | None:
    """Read selected_pair from the config file. Returns None for auto-selection."""
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pair = cfg.get("ardl", {}).get("selected_pair", None)
    if pair is not None:
        return (int(pair[0]), int(pair[1]))
    return None  # triggers auto-selection in step_05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the ARDL forecasting experiment")
    p.add_argument("--config", default="configs/config.yaml")
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config

    selected_pair = _load_selected_pair(config_path)
    if selected_pair is not None:
        print(f"[ARDL] Using fixed selected_pair from config: {selected_pair}")
        context = {"SELECTED_PAIR": selected_pair}
    else:
        print("[ARDL] selected_pair=null in config → auto-selection after sweep (BIC)")
        context = {}

    for step in [
        step_01_setup,
        step_02_find_project_root,
        step_03_load_data,
        step_04_validate_pca,
        step_04a_adf_stationarity_test,
        step_05_sweep_ardl,
        step_06_select_and_forecast,
        step_07_export_pkl,
        step_08_summary,
        step_09_plot,
        step_10_ardl_80obs,
        step_11_summary_table,
    ]:
        context = step(context)

    return context


if __name__ == "__main__":
    main()
