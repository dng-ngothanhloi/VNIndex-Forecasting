"""
test_split_boundary.py – Split_Boundary unit + property tests
==================================================================
Feature: base-preprocessor-wrapper

Covers:
- Task 6.3: unit test - non-overlap branch equivalence
- Task 6.4: unit test - overlap branch equivalence
- Task 6.5: Property 6 (Split_Boundary equivalent to direct split call)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
from hypothesis import given, settings, strategies as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.utils import split_by_time  # noqa: E402
from src.preprocessing.split_utils import split_time_series_overlap_val  # noqa: E402
from src.preprocessing.split_boundary import build_split_boundary  # noqa: E402


def _boundaries_equal(a, b) -> bool:
    """Compare two boundary values treating NaT == NaT as equal.

    Empty splits (possible for small n_obs / small ratios) yield NaT from
    .index.min()/.max(); NaT == NaT is False by pandas/NumPy convention, so
    a plain == would spuriously fail even when both sides are identically
    NaT. This purely affects test comparison logic, not build_split_boundary
    itself, which reproduces the underlying split function's behavior as-is.
    """
    if pd.isna(a) and pd.isna(b):
        return True
    return a == b


def _make_synthetic_wide_df(n_obs: int = 200, n_features: int = 5, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    cols = [f"STOCK_{i}" for i in range(n_features)]
    return pd.DataFrame(rng.normal(size=(n_obs, n_features)), index=dates, columns=cols)


_FIXED_SYNTHETIC_DF = _make_synthetic_wide_df()


# ─────────────────────────────────────────────────────────────────
# Task 6.3: unit test - non-overlap branch equivalence
# ─────────────────────────────────────────────────────────────────
def test_build_split_boundary_non_overlap_matches_direct_call():
    df = _FIXED_SYNTHETIC_DF
    prep_cfg = {"use_overlap_val": False, "train_ratio": 0.7, "val_ratio": 0.15}
    boundary = build_split_boundary(df, prep_cfg)
    ref = split_by_time(df, train_ratio=0.7, val_ratio=0.15)

    pdt.assert_frame_equal(boundary.train, ref.train)
    pdt.assert_frame_equal(boundary.val, ref.val)
    pdt.assert_frame_equal(boundary.test, ref.test)
    assert _boundaries_equal(boundary.train_start, ref.train.index.min())
    assert _boundaries_equal(boundary.train_end, ref.train.index.max())
    assert _boundaries_equal(boundary.val_start, ref.val.index.min())
    assert _boundaries_equal(boundary.val_end, ref.val.index.max())
    assert _boundaries_equal(boundary.test_start, ref.test.index.min())
    assert _boundaries_equal(boundary.test_end, ref.test.index.max())


# ─────────────────────────────────────────────────────────────────
# Task 6.4: unit test - overlap branch equivalence
# ─────────────────────────────────────────────────────────────────
def test_build_split_boundary_overlap_matches_direct_call():
    df = _FIXED_SYNTHETIC_DF
    prep_cfg = {"use_overlap_val": True, "train_ratio": 0.7, "val_overlap_ratio": 0.20}
    boundary = build_split_boundary(df, prep_cfg)
    ref = split_time_series_overlap_val(df, train_ratio=0.7, val_overlap_ratio=0.20)

    pdt.assert_frame_equal(boundary.train, ref.train)
    pdt.assert_frame_equal(boundary.val, ref.val)
    pdt.assert_frame_equal(boundary.test, ref.test)
    assert _boundaries_equal(boundary.train_start, ref.train.index.min())
    assert _boundaries_equal(boundary.train_end, ref.train.index.max())


# ─────────────────────────────────────────────────────────────────
# Task 6.5 / Property 6: Split_Boundary equivalent to direct split call
# **Property 6: Split_Boundary is equivalent to a direct call to the selected split function**
# **Validates: Requirements 5.2, 5.3, 5.4, 5.5**
# ─────────────────────────────────────────────────────────────────
@given(
    n_obs=st.integers(min_value=10, max_value=200),
    n_features=st.integers(min_value=1, max_value=5),
    use_overlap_val=st.booleans(),
    train_ratio=st.floats(min_value=0.5, max_value=0.8),
    val_ratio=st.floats(min_value=0.05, max_value=0.2),
    val_overlap_ratio=st.floats(min_value=0.05, max_value=0.3),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=100)
def test_build_split_boundary_matches_direct_call_property(
    n_obs, n_features, use_overlap_val, train_ratio, val_ratio, val_overlap_ratio, seed
):
    df = _make_synthetic_wide_df(n_obs=n_obs, n_features=n_features, seed=seed)
    prep_cfg = {
        "use_overlap_val": use_overlap_val,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "val_overlap_ratio": val_overlap_ratio,
    }

    boundary = build_split_boundary(df, prep_cfg)

    if use_overlap_val:
        ref = split_time_series_overlap_val(
            df, train_ratio=train_ratio, val_overlap_ratio=val_overlap_ratio
        )
    else:
        ref = split_by_time(df, train_ratio=train_ratio, val_ratio=val_ratio)

    pdt.assert_frame_equal(boundary.train, ref.train)
    pdt.assert_frame_equal(boundary.val, ref.val)
    pdt.assert_frame_equal(boundary.test, ref.test)
    assert _boundaries_equal(boundary.train_start, ref.train.index.min())
    assert _boundaries_equal(boundary.train_end, ref.train.index.max())
    assert _boundaries_equal(boundary.val_start, ref.val.index.min())
    assert _boundaries_equal(boundary.val_end, ref.val.index.max())
    assert _boundaries_equal(boundary.test_start, ref.test.index.min())
    assert _boundaries_equal(boundary.test_end, ref.test.index.max())
