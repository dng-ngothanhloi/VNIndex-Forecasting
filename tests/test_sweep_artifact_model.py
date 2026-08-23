"""
test_sweep_artifact_model.py – Tests for grouped sweep artifact structure
==========================================================================
Validates the new artifact schema where ONE full-sweep produces ONE parent
Run_* with child label subdirectories.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.compare_representations import collect_from_run_dir


def _make_child(parent: Path, label: str, method: str, cev=None, status="OK",
                parent_run_id=None, k_actual=None, p_original=318):
    """Create a minimal valid child directory with manifest + summary."""
    child = parent / label
    child.mkdir(parents=True)
    (child / "results").mkdir(parents=True)

    if parent_run_id is None:
        parent_run_id = parent.name

    repr_info = {
        "method": method,
        "cev_requested": cev,
        "cev_achieved": cev * 1.01 if cev else None,
        "p_original": p_original,
        "k_actual": k_actual if k_actual else (p_original if method == "none" else 2),
        "dim_reduction_pct": 0.0 if method == "none" else 99.0,
    }

    manifest = {
        "parent_run_id": parent_run_id,
        "label": label,
        "status": status,
        "representation": repr_info,
    }
    with open(child / "run_manifest.json", "w") as f:
        json.dump(manifest, f)

    summary = {
        "label": label,
        "status": status,
        "representation": repr_info,
        "ardl": {"ARDL_RMSE": 14.0, "ARDL_P": 1, "ARDL_Q": 1, "ARDL_N": 167},
        "lstm": {"LSTM_RMSE": 25.0, "LSTM_LB": 60, "LSTM_BS": 32, "LSTM_N": 167},
        "dm": {"DM_MSE_p": 0.05, "DM_MAE_p": 0.001},
    }
    with open(child / "results" / "run_summary.json", "w") as f:
        json.dump(summary, f)


def _make_sweep_manifest(parent: Path, labels: list[str], completed=None, failed=None):
    manifest = {
        "schema_version": 2,
        "run_id": parent.name,
        "experiment_type": "representation_sweep",
        "planned_labels": labels,
        "completed_labels": completed or labels,
        "failed_labels": failed or [],
        "status": "OK",
    }
    with open(parent / "sweep_manifest.json", "w") as f:
        json.dump(manifest, f)


# ─────────────────────────────────────────────────────────────────
# 1. Full sweep creates ONE parent Run_*
# ─────────────────────────────────────────────────────────────────
def test_full_sweep_one_parent(tmp_path):
    parent = tmp_path / "Run_20260811_120000"
    parent.mkdir()
    labels = ["no_dr", "pca_cev_0.75", "pca_cev_0.85"]
    _make_sweep_manifest(parent, labels)
    _make_child(parent, "no_dr", "none")
    _make_child(parent, "pca_cev_0.75", "pca", cev=0.75)
    _make_child(parent, "pca_cev_0.85", "pca", cev=0.85)

    df = collect_from_run_dir(parent)
    assert len(df) == 3
    assert set(df["label"]) == set(labels)


# ─────────────────────────────────────────────────────────────────
# 2. Child labels are deterministic
# ─────────────────────────────────────────────────────────────────
def test_child_labels_deterministic():
    from experiments.run_experiment import _label_for
    assert _label_for("none", None) == "no_dr"
    assert _label_for("pca", 0.75) == "pca_cev_0.75"
    assert _label_for("pca", 0.80) == "pca_cev_0.80"
    assert _label_for("pca", 0.95) == "pca_cev_0.95"


# ─────────────────────────────────────────────────────────────────
# 4. Each child manifest has matching parent_run_id
# ─────────────────────────────────────────────────────────────────
def test_child_parent_run_id_mismatch_excluded(tmp_path):
    parent = tmp_path / "Run_20260811_120000"
    parent.mkdir()
    _make_sweep_manifest(parent, ["no_dr", "pca_cev_0.75"])
    _make_child(parent, "no_dr", "none")
    _make_child(parent, "pca_cev_0.75", "pca", cev=0.75,
                parent_run_id="Run_DIFFERENT")

    df = collect_from_run_dir(parent)
    assert len(df) == 1  # only no_dr included
    assert df.iloc[0]["label"] == "no_dr"


# ─────────────────────────────────────────────────────────────────
# 5. compare_representations requires --run-dir
# ─────────────────────────────────────────────────────────────────
def test_compare_requires_run_dir():
    """CLI parser requires --run-dir argument."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "src.evaluation.compare_representations"],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "required" in result.stderr.lower() or "run-dir" in result.stderr.lower()


# ─────────────────────────────────────────────────────────────────
# 6. Comparator reads only supplied Run_*
# ─────────────────────────────────────────────────────────────────
def test_comparator_reads_only_supplied_dir(tmp_path):
    # Create two parents
    parent1 = tmp_path / "Run_A"
    parent1.mkdir()
    _make_sweep_manifest(parent1, ["no_dr"])
    _make_child(parent1, "no_dr", "none")

    parent2 = tmp_path / "Run_B"
    parent2.mkdir()
    _make_sweep_manifest(parent2, ["pca_cev_0.85"])
    _make_child(parent2, "pca_cev_0.85", "pca", cev=0.85)

    # Reading parent1 should NOT include parent2's children
    df = collect_from_run_dir(parent1)
    assert len(df) == 1
    assert df.iloc[0]["label"] == "no_dr"


# ─────────────────────────────────────────────────────────────────
# 7. Another Run_* cannot affect aggregation
# ─────────────────────────────────────────────────────────────────
def test_newer_run_does_not_affect_older(tmp_path):
    parent_old = tmp_path / "Run_20260811_100000"
    parent_old.mkdir()
    _make_sweep_manifest(parent_old, ["no_dr"])
    _make_child(parent_old, "no_dr", "none")

    parent_new = tmp_path / "Run_20260811_200000"
    parent_new.mkdir()
    _make_sweep_manifest(parent_new, ["pca_cev_0.90"])
    _make_child(parent_new, "pca_cev_0.90", "pca", cev=0.90)

    df = collect_from_run_dir(parent_old)
    assert len(df) == 1
    assert "pca_cev_0.90" not in df["label"].values


# ─────────────────────────────────────────────────────────────────
# 8. Duplicate labels inside one Run_* cause failure
# ─────────────────────────────────────────────────────────────────
def test_duplicate_labels_cause_failure(tmp_path):
    parent = tmp_path / "Run_20260811_120000"
    parent.mkdir()
    _make_sweep_manifest(parent, ["no_dr"])
    _make_child(parent, "no_dr", "none")
    # Create a second dir with same label in manifest
    dup = parent / "no_dr_copy"
    dup.mkdir()
    (dup / "results").mkdir()
    manifest = {"parent_run_id": parent.name, "label": "no_dr", "status": "OK",
                "representation": {"method": "none", "cev_requested": None,
                                   "p_original": 318, "k_actual": 318, "dim_reduction_pct": 0.0}}
    with open(dup / "run_manifest.json", "w") as f:
        json.dump(manifest, f)
    summary = {"label": "no_dr", "status": "OK",
               "representation": manifest["representation"],
               "ardl": {}, "lstm": {}, "dm": {}}
    with open(dup / "results" / "run_summary.json", "w") as f:
        json.dump(summary, f)

    with pytest.raises(SystemExit):
        collect_from_run_dir(parent)


# ─────────────────────────────────────────────────────────────────
# 9. PCA label/manifest/summary CEV mismatch causes exclusion
# ─────────────────────────────────────────────────────────────────
def test_pca_cev_mismatch_excluded(tmp_path):
    parent = tmp_path / "Run_20260811_120000"
    parent.mkdir()
    _make_sweep_manifest(parent, ["pca_cev_0.75"])

    # Create child with mismatched label vs manifest cev
    child = parent / "pca_cev_0.75"
    child.mkdir()
    (child / "results").mkdir()
    manifest = {"parent_run_id": parent.name, "label": "pca_cev_0.75", "status": "OK",
                "representation": {"method": "pca", "cev_requested": 0.85,  # MISMATCH
                                   "p_original": 318, "k_actual": 4, "dim_reduction_pct": 98.7}}
    with open(child / "run_manifest.json", "w") as f:
        json.dump(manifest, f)
    summary = {"label": "pca_cev_0.75", "status": "OK",
               "representation": {"method": "pca", "cev_requested": 0.85,
                                   "p_original": 318, "k_actual": 4, "dim_reduction_pct": 98.7},
               "ardl": {}, "lstm": {}, "dm": {}}
    with open(child / "results" / "run_summary.json", "w") as f:
        json.dump(summary, f)

    df = collect_from_run_dir(parent)
    assert len(df) == 0  # excluded due to mismatch


# ─────────────────────────────────────────────────────────────────
# 10. NoReduction cannot contain PCA k semantics
# ─────────────────────────────────────────────────────────────────
def test_noreduction_with_pca_k_excluded(tmp_path):
    parent = tmp_path / "Run_20260811_120000"
    parent.mkdir()
    _make_sweep_manifest(parent, ["no_dr"])

    # NoReduction child with k_actual != p_original (invalid)
    _make_child(parent, "no_dr", "none", k_actual=5, p_original=318)

    df = collect_from_run_dir(parent)
    assert len(df) == 0  # excluded


# ─────────────────────────────────────────────────────────────────
# 11. Comparison output saved under <run-dir>/comparison/
# ─────────────────────────────────────────────────────────────────
def test_comparison_saved_under_run_dir(tmp_path):
    from src.evaluation.compare_representations import save_comparison

    parent = tmp_path / "Run_20260811_120000"
    parent.mkdir()

    df = pd.DataFrame([{"label": "no_dr", "ARDL_RMSE": 14.0}])
    save_comparison(df, parent)

    assert (parent / "comparison" / "representation_comparison.csv").exists()
    assert (parent / "comparison" / "representation_comparison.json").exists()


# ─────────────────────────────────────────────────────────────────
# 12. Legacy global "latest per CEV" logic removed
# ─────────────────────────────────────────────────────────────────
def test_no_global_artifact_scanning():
    """compare_representations does not have a mode that scans all Run_* globally."""
    src = (PROJECT_ROOT / "src" / "evaluation" / "compare_representations.py").read_text()
    # Should not contain glob("Run_*") pattern scanning
    assert 'glob("Run_*")' not in src
    assert "artifacts_dir.glob" not in src


# ─────────────────────────────────────────────────────────────────
# 13. --include-multiseed flag exists and threads into _run_child
# ─────────────────────────────────────────────────────────────────
def test_include_multiseed_flag_exists():
    from experiments.run_experiment import parse_args
    args = parse_args(["--full-sweep", "--include-multiseed"])
    assert args.include_multiseed is True
    args_off = parse_args(["--full-sweep"])
    assert args_off.include_multiseed is False


def test_run_child_accepts_include_multiseed_kwarg():
    """_run_child must accept include_multiseed so the sweep can thread it."""
    import inspect
    from experiments.run_experiment import _run_child
    sig = inspect.signature(_run_child)
    assert "include_multiseed" in sig.parameters
    # Default must be False so existing callers are unaffected
    assert sig.parameters["include_multiseed"].default is False


def test_multiseed_dir_is_snapshotted_into_child():
    """The multiseed output dir must be in the snapshot list so results land
    in <Run_*>/<label>/outputs/lstm_vnindex_multiseed/."""
    from experiments.run_experiment import _MUTABLE_OUTPUT_DIRS
    assert "outputs/lstm_vnindex_multiseed" in _MUTABLE_OUTPUT_DIRS


def test_non_fatal_multiseed_failure_keeps_child_ok(tmp_path):
    """A child whose only failed step is MultiSeed(non-fatal) must still be
    aggregated (status OK), because ARDL/LSTM/DM results remain valid."""
    parent = tmp_path / "Run_20260811_120000"
    parent.mkdir()
    _make_sweep_manifest(parent, ["pca_cev_0.85"])
    _make_child(parent, "pca_cev_0.85", "pca", cev=0.85, status="OK")

    df = collect_from_run_dir(parent)
    assert len(df) == 1
    assert df.iloc[0]["label"] == "pca_cev_0.85"
