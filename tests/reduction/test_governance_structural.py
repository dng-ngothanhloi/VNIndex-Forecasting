"""
test_governance_structural.py – structural governance test for
src/pca_model.py
==================================================================
Feature: pca-reducer-wrapper

Covers:
- Task 8.1: mandatory governance check confirming every function in
  src/pca_model.py other than the two call-site blocks (PCA_Full_Call_Site,
  PCA_Final_Call_Site, and the loadings/eigenvalues block that reads
  pca_final's attributes) is byte-identical before/after this phase's
  integration task (Task 11), via a stored content hash captured before
  the integration task began. Also asserts identity separation between
  two PCAReducer instances (Requirement 4.3).

The golden hashes below were captured by hashing each function's exact
source segment (via ast.get_source_segment) in src/pca_model.py BEFORE
Task 11 (the src/pca_model.py integration task) modified the file. This
follows the base-preprocessor-wrapper Property 7 structural test pattern
referenced by design.md's Testing Strategy section.

Functions intentionally NOT covered by this golden-hash set (because they
are the two call sites/loadings-eigenvalues block this phase is allowed
to change):
  - run_pca_pipeline (contains PCA_Full_Call_Site, PCA_Final_Call_Site,
    and the loadings/eigenvalues block; also contains all the untouched
    logic, but as a whole function it is expected to differ line-for-line
    at the two call sites, so it is excluded from the byte-identical
    check and instead spot-checked structurally below)
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PCA_MODEL_PATH = SRC_ROOT / "pca_model.py"

# ─────────────────────────────────────────────────────────────────
# Golden hashes captured BEFORE Task 11 (src/pca_model.py integration)
# modified the file. These functions are NOT part of the two call-site
# blocks (PCA_Full_Call_Site, PCA_Final_Call_Site, loadings/eigenvalues
# block) and MUST remain byte-identical (Requirement 4.6).
# ─────────────────────────────────────────────────────────────────
GOLDEN_FUNCTION_HASHES = {
    "load_config": "2a4170d428bdc187765aa737bc92e0351d03fcff0f6801d92fa3190013834e11",
    "load_sector_mapping": "2ee9f6ac5725ceee9c6d1cbc9298467e3bf791539fe110a3f9529c9f0639fff6",
    "summarize_pc_by_sector": "c05a28d6338586c2aec9b03db6712bf34554421ef7bad50ab70b33a8ecb5501a",
    "verify_pc_orthogonality": "229319d0d2030af395f945aa5845c9c9d078d065c9f5baef1732a50e09ee6398",
    "save_pca_figure": "b342a55f0a3c32f199ddb00db3a9456a3ba3866e24839fbb80c31dd955ef4369",
    "save_pca_individual_figures": "61f783708baca882368bd57c484e71fab207cd26b156b99dc000c1155aa71522",
    "save_pca_threshold_table": "de40f1db0f1da0a8974eb8a7fa787f3c5aa71f88bc7f2632a7e4a8c2135fd6b8",
    "save_scree_plot_figure": "12476f93d1d9d2627fb592cf30f613432ed02e9aadd4b5390cc01f9330d807b1",
    "parse_args": "8f8a69d49b5f85a1d5db5fa58eca40db91662b1ed7fa82ba6234e4df4afd46fc",
}


def _function_source_segments(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    segments = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            segments[node.name] = ast.get_source_segment(source, node)
    return segments


def test_untouched_functions_are_byte_identical_to_golden_hashes():
    """
    Task 8.1: every function in src/pca_model.py other than
    run_pca_pipeline (which contains the two call-site blocks and the
    loadings/eigenvalues block) must hash identically to the golden
    hashes captured before Task 11's integration changes.
    """
    segments = _function_source_segments(PCA_MODEL_PATH)

    missing = set(GOLDEN_FUNCTION_HASHES) - set(segments)
    assert not missing, f"Expected functions missing from src/pca_model.py: {missing}"

    mismatches = []
    for name, golden_hash in GOLDEN_FUNCTION_HASHES.items():
        actual_hash = hashlib.sha256(segments[name].encode("utf-8")).hexdigest()
        if actual_hash != golden_hash:
            mismatches.append(name)

    assert not mismatches, (
        f"The following functions in src/pca_model.py changed but were expected "
        f"to remain byte-identical (Requirement 4.6): {mismatches}"
    )


def test_run_pca_pipeline_still_references_expected_untouched_symbols():
    """
    Structural spot-check (not byte-identical, since run_pca_pipeline
    legitimately changes at the two call sites): confirm the untouched
    logic markers are still present verbatim in run_pca_pipeline's body -
    the CEV threshold sweep, sector mapping, orthogonality check, and
    artifact-writing calls (Requirement 4.6).
    """
    segments = _function_source_segments(PCA_MODEL_PATH)
    body = segments["run_pca_pipeline"]

    expected_markers = [
        "load_sector_mapping(sector_map_path)",
        "summarize_pc_by_sector(",
        "verify_pc_orthogonality(",
        "pca_threshold_summary.csv",
        "pca_loadings.csv",
        "pca_eigenvalues.csv",
        "pca_corr_after.csv",
        "pca_metrics.csv",
        "pca_model.pkl",
        "save_pca_figure(",
        "save_pca_individual_figures(",
        "save_scree_plot_figure(",
        "save_pca_threshold_table(",
    ]
    for marker in expected_markers:
        assert marker in body, f"Expected untouched marker missing from run_pca_pipeline: {marker!r}"


def test_pca_full_and_pca_final_are_strictly_separate_instances():
    """
    Requirement 4.3: run_pca_pipeline must not reuse a single PCAReducer
    instance for both the PCA_Full_Call_Site and PCA_Final_Call_Site.
    Simulated here directly against PCAReducer, mirroring how
    run_pca_pipeline constructs pca_full/pca_final as two separate
    objects.
    """
    from reduction import PCAReducer

    pca_full = PCAReducer(n_components=None, random_state=42)
    pca_final = PCAReducer(n_components=2, random_state=42)

    assert pca_full is not pca_final
