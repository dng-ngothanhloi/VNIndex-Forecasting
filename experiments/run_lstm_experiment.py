#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_lstm_experiment.py – LSTM experiment entry point (canonical)
======================================================================
Runs the scientifically-corrected LSTM lookback x batch_size sweep,
cross-boundary Val/Test windowing, and final Train+Val refit pipeline
(src/forecasting/lstm/*) end to end. This is the canonical replacement for
the former lstm/run_all_lstm.py.

Usage:
    python experiments/run_lstm_experiment.py
"""

import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.lstm.setup import run_imports, run_paths
from src.forecasting.lstm.data import run_load_data, run_prepare_data
from src.forecasting.lstm.sweep import run_train_and_evaluate
from src.forecasting.lstm.export import run_export_model
from src.forecasting.lstm.reporting import run_model_summary, run_export_figures


def print_header(text: str, width: int = 80):
    print("\n" + "=" * width)
    print(f" {text} ".center(width))
    print("=" * width)


def run_step(step_num: int, step_name: str, step_func, context: dict):
    """Run one step, threading the shared context dict explicitly through
    every step (explicit state passing -- no cross-module import side
    effects)."""
    print_header(f"Step {step_num:02d}: {step_name}")
    print(f"Executing {step_func.__module__}.{step_func.__name__}...\n")

    start_time = time.time()
    try:
        context = step_func(context)
        elapsed = time.time() - start_time
        print(f"\n[PASS] Step {step_num} completed successfully in {elapsed:.2f} seconds")
        return True, context
    except FileNotFoundError as e:
        print(f"\n[FAIL] ERROR: File not found - {e}")
    except AssertionError as e:
        print(f"\n[FAIL] ERROR: Assertion failed - {e}")
    except KeyboardInterrupt:
        print(f"\n[WARN] Step {step_num} interrupted by user.")
        raise
    except Exception as e:
        print(f"\n[FAIL] ERROR: {type(e).__name__} - {e}")
        traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n[FAIL] Step {step_num} failed after {elapsed:.2f} seconds")
    return False, context


def main() -> dict:
    print_header("LSTM VNINDEX PREDICTION EXPERIMENT")

    steps = [
        (1, "Imports and environment setup", run_imports),
        (2, "Path configuration", run_paths),
        (3, "Load PCA features and VNINDEX target", run_load_data),
        (4, "Prepare data - scaling and windowing", run_prepare_data),
        (5, "Train and evaluate LSTM models", run_train_and_evaluate),
        (6, "Export the selected LSTM model", run_export_model),
        (7, "Generate LSTM model summary report", run_model_summary),
        (8, "Export figures to logs/figures/lstm", run_export_figures),
    ]

    total_start = time.time()
    completed_steps, failed_steps = [], []
    context: dict = {}

    for step_num, step_name, step_func in steps:
        success, context = run_step(step_num, step_name, step_func, context)
        if success:
            completed_steps.append(step_num)
        else:
            failed_steps.append(step_num)
            print(f"\n[WARN] Pipeline stopped due to failure at Step {step_num}: {step_name}")
            break

    total_elapsed = time.time() - total_start
    print_header("EXECUTION SUMMARY")
    print(f"Total execution time: {total_elapsed:.2f} seconds")
    print(f"Steps completed: {len(completed_steps)}/{len(steps)}")
    if failed_steps:
        print(f"Failed steps: {', '.join(str(s) for s in failed_steps)}")
        sys.exit(1)
    else:
        print("[COMP] All steps completed successfully!")

    return context


if __name__ == "__main__":
    main()
