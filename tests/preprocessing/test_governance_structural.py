"""
test_governance_structural.py – structural governance checks
==================================================================
Feature: base-preprocessor-wrapper (Property 8)

ROLLBACK NOTE: the Stage-2 leakage-safety correction (which deleted
step_3_outliers.py / step_5_filter_observation_ratio.py /
step_6_fill_and_clean.py in favor of src/preprocessing/standard.py's
Train-only fitting) has been diagnostically rolled back. These three
files are restored and back on the active preprocessing path with their
original full-date-range behavior.

Property 7 ("Critical-flagged pre-split step files remain byte-identical")
is checked here against the CURRENT restored files rather than a
pre-Stage-1 byte hash: the Stage-1 architecture migration legitimately
changed these files' import statement (`helpers.utils` -> `src.utils`),
so a strict byte-identity check against the original pre-migration hash
would fail for a reason unrelated to Stage 2. Instead, this test asserts
the three files exist and still contain their original algorithmic
behavior markers (function signatures unchanged).

Property 8 (governed artifact paths remain stable) is restored to the
ORIGINAL pre-Stage-2 artifact schema: quality/outlier_log.csv (not
outlier_bounds.csv), and no standard_preprocessor_params.pkl.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ─────────────────────────────────────────────────────────────────
# Property 7 (rollback-adjusted): the three pre-split step files must
# be PRESENT and retain their original function signatures/behavior --
# not a byte-identity check, since the Stage-1 architecture migration
# legitimately changed their import line (helpers.utils -> src.utils).
# ─────────────────────────────────────────────────────────────────
_CRITICAL_FILES_EXPECTED_SIGNATURES = {
    "src/preprocess_steps/step_3_outliers.py": "def remove_outliers_with_log(",
    "src/preprocess_steps/step_5_filter_observation_ratio.py": "def filter_by_observation_ratio(",
    "src/preprocess_steps/step_6_fill_and_clean.py": "def fill_and_clean(",
}


def test_critical_flagged_pre_split_step_files_are_restored_and_active():
    for rel_path, expected_signature in _CRITICAL_FILES_EXPECTED_SIGNATURES.items():
        abs_path = PROJECT_ROOT / rel_path
        assert abs_path.exists(), f"Expected restored critical file missing: {rel_path}"
        content = abs_path.read_text(encoding="utf-8")
        assert expected_signature in content, (
            f"{rel_path} does not contain the expected original function "
            f"signature '{expected_signature}' -- rollback may be incomplete."
        )
        # Original full-date-range fill behavior marker: ffill().bfill()
        # must still be present in step_6 (never removed for this rollback).
    fill_src = (PROJECT_ROOT / "src/preprocess_steps/step_6_fill_and_clean.py").read_text(encoding="utf-8")
    assert "ffill().bfill()" in fill_src, "Original forward/backward-fill behavior must be restored exactly."


def test_preprocess_pipeline_calls_restored_pre_split_steps_not_standard_preprocessor():
    preprocess_source = (SRC_ROOT / "preprocess.py").read_text(encoding="utf-8")
    assert "remove_outliers_with_log" in preprocess_source
    assert "filter_by_observation_ratio" in preprocess_source
    assert "fill_and_clean" in preprocess_source
    # Check ACTUAL usage (import statement / instantiation), not prose
    # mentions in comments/docstrings explaining the rollback.
    assert "from preprocessing import" in preprocess_source
    import_line = next(
        line for line in preprocess_source.splitlines() if line.strip().startswith("from preprocessing import")
    )
    assert "StandardPreprocessor" not in import_line, (
        "StandardPreprocessor must not be imported/used on the active "
        "preprocessing path after the Stage-2 diagnostic rollback."
    )
    assert "StandardPreprocessor(" not in preprocess_source, (
        "StandardPreprocessor must not be instantiated on the active "
        "preprocessing path after the Stage-2 diagnostic rollback."
    )


# ─────────────────────────────────────────────────────────────────
# Property 8 (restored to original pre-Stage-2 schema): Every governed
# artifact path remains stable across pipeline runs.
# **Property 8: Every governed artifact path remains stable across pipeline runs**
# **Validates: Requirements 7.2, 9.3**
# ─────────────────────────────────────────────────────────────────
_GOVERNED_ARTIFACT_RELATIVE_PATHS = [
    "core/cleaned_data.csv",
    "core/vnindex_target.csv",
    "core/valid_stocks.csv",
    "core/removed_stocks.csv",
    "quality/outlier_log.csv",
    "quality/missing_dist.csv",
    "quality/corr_summary.csv",
    "quality/corr_matrix.csv",
    "splits/split_summary.csv",
    "splits/train_scaled.csv",
    "splits/val_scaled.csv",
    "splits/test_scaled.csv",
    "stationarity/stationarity_results.csv",
    "stationarity/stationarity_logreturn.csv",
    "models/scaler_params.pkl",
]


def test_governed_artifact_filenames_referenced_in_preprocess_pipeline():
    preprocess_source = (SRC_ROOT / "preprocess.py").read_text(encoding="utf-8")

    for rel_path in _GOVERNED_ARTIFACT_RELATIVE_PATHS:
        filename = rel_path.split("/")[-1]
        assert filename in preprocess_source, (
            f"Expected artifact filename '{filename}' (from governed path '{rel_path}') "
            "to still be referenced by src/preprocess.py's Pipeline_Shell."
        )
