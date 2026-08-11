"""
test_preprocess_pipeline_smoke.py – Real_Data_Smoke_Test
=============================================================
Feature: base-preprocessor-wrapper (Task 10.1)

Runs preprocess_pipeline() end-to-end against the actual raw CSV
referenced by config/config.yaml. Skips gracefully (never fails) when
that raw file is not present in the execution environment, per
Requirement 8.6.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from preprocess import preprocess_pipeline, load_config  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _raw_file_exists() -> bool:
    cfg = load_config(CONFIG_PATH)
    return (PROJECT_ROOT / cfg["paths"]["raw_file"]).exists()


@pytest.mark.skipif(not _raw_file_exists(), reason="raw data file not present in this environment")
def test_preprocess_pipeline_end_to_end_smoke():
    preprocess_pipeline(project_root=PROJECT_ROOT, config_path=CONFIG_PATH)

    cfg = load_config(CONFIG_PATH)
    paths = cfg["paths"]
    processed_dir = PROJECT_ROOT / paths.get("artifacts_dir", paths.get("processed_dir", "data/processed"))
    subdirs = paths.get("artifacts_subdirs", paths.get("processed_subdirs", {}))
    splits_dir = processed_dir / subdirs.get("splits", "splits")

    train_scaled = pd.read_csv(splits_dir / "train_scaled.csv", index_col=0)
    assert abs(train_scaled.mean().mean()) < 1e-6
    assert abs(train_scaled.std().mean() - 1.0) < 1e-2
