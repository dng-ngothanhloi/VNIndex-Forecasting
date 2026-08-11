"""
test_zscore_preprocessor.py – ZScorePreprocessor unit + property tests
==========================================================================
Feature: base-preprocessor-wrapper

Covers:
- Task 4.2: unit test for ZScorePreprocessor public API surface
- Task 4.3: Property 3 (public surface never grows)
- Task 4.4: Property 5 (NotFittedError before fit)
- Task 4.5: Property 4 (numerical equivalence with scale_by_train_stats)
- Task 8.1: Synthetic_Parity_Fixture regression test
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.utils import scale_by_train_stats  # noqa: E402
from src.preprocessing.zscore import ZScorePreprocessor  # noqa: E402
from src.preprocessing.base import NotFittedError  # noqa: E402

_EXPECTED_PUBLIC_API = {"fit", "transform", "fit_transform", "get_metadata"}


def _public_attrs(obj) -> set:
    return {name for name in dir(obj) if not name.startswith("_")}


# ─────────────────────────────────────────────────────────────────
# Task 4.2: unit test - public API surface is exactly the 4 methods
# ─────────────────────────────────────────────────────────────────
def test_zscore_preprocessor_public_api_is_minimal():
    p = ZScorePreprocessor()
    assert _public_attrs(p) == _EXPECTED_PUBLIC_API


# ─────────────────────────────────────────────────────────────────
# Task 4.3 / Property 3: ZScorePreprocessor's public surface never grows
# **Property 3: ZScorePreprocessor's public surface never grows**
# **Validates: Requirements 2.2**
# ─────────────────────────────────────────────────────────────────
@given(
    n_obs=st.integers(min_value=5, max_value=50),
    n_features=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_zscore_preprocessor_public_surface_never_grows(n_obs, n_features):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.normal(size=(n_obs, n_features)),
        columns=[f"col_{i}" for i in range(n_features)],
    )

    p = ZScorePreprocessor()
    assert _public_attrs(p) == _EXPECTED_PUBLIC_API

    p.fit(df)
    assert _public_attrs(p) == _EXPECTED_PUBLIC_API

    p.transform(df)
    assert _public_attrs(p) == _EXPECTED_PUBLIC_API


# ─────────────────────────────────────────────────────────────────
# Task 4.4 / Property 5: Methods requiring fit raise NotFittedError
# **Property 5: Methods requiring fit raise NotFittedError before fit is called**
# **Validates: Requirements 2.5, 3.5**
# ─────────────────────────────────────────────────────────────────
@given(
    n_obs=st.integers(min_value=1, max_value=20),
    n_features=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=100)
def test_unfitted_methods_raise_not_fitted_error(n_obs, n_features):
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        rng.normal(size=(n_obs, n_features)),
        columns=[f"col_{i}" for i in range(n_features)],
    )

    p = ZScorePreprocessor()

    with pytest.raises(NotFittedError):
        p.transform(df)

    with pytest.raises(NotFittedError):
        p.get_metadata()


# ─────────────────────────────────────────────────────────────────
# Task 4.5 / Property 4: numerical equivalence with scale_by_train_stats
# **Property 4: ZScorePreprocessor is numerically equivalent to scale_by_train_stats**
# **Validates: Requirements 2.3, 2.4, 2.6, 3.1, 3.2, 3.3, 3.4, 7.6, 8.3, 8.4**
# ─────────────────────────────────────────────────────────────────
@given(
    n_train=st.integers(min_value=5, max_value=50),
    n_val=st.integers(min_value=1, max_value=20),
    n_test=st.integers(min_value=1, max_value=20),
    n_features=st.integers(min_value=1, max_value=10),
    include_zero_variance_col=st.booleans(),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=100)
def test_zscore_preprocessor_matches_scale_by_train_stats_property(
    n_train, n_val, n_test, n_features, include_zero_variance_col, seed
):
    rng = np.random.default_rng(seed)
    cols = [f"col_{i}" for i in range(n_features)]

    train = pd.DataFrame(rng.normal(size=(n_train, n_features)), columns=cols)
    val = pd.DataFrame(rng.normal(size=(n_val, n_features)), columns=cols)
    test = pd.DataFrame(rng.normal(size=(n_test, n_features)), columns=cols)

    if include_zero_variance_col:
        # Exercise the zero-std-replaced-by-1 path.
        train[cols[0]] = 1.0

    train_ref, val_ref, test_ref, mean_ref, std_ref = scale_by_train_stats(train, val, test)

    p = ZScorePreprocessor()
    p.fit(train)
    train_wrapped = p.transform(train)
    val_wrapped = p.transform(val)
    test_wrapped = p.transform(test)
    meta = p.get_metadata()

    np.testing.assert_allclose(train_wrapped.values, train_ref.values, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(val_wrapped.values, val_ref.values, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(test_wrapped.values, test_ref.values, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(meta["train_mean"].values, mean_ref.values, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(meta["train_std"].values, std_ref.values, rtol=1e-12, atol=1e-15)
    assert meta["feature_names"] == train.columns.tolist()


# ─────────────────────────────────────────────────────────────────
# Task 8.1: Synthetic_Parity_Fixture regression test (non-optional)
# ─────────────────────────────────────────────────────────────────
def make_synthetic_wide_df(rng: np.random.Generator, n_obs: int = 500, n_features: int = 40) -> pd.DataFrame:
    """Shape chosen for fast CI runs; illustrative, not tied to the real
    ~(578, 318) train matrix shape referenced in pca-reducer-wrapper's design."""
    dates = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    cols = [f"STOCK_{i}" for i in range(n_features)]
    return pd.DataFrame(rng.normal(0.0, 1.0, size=(n_obs, n_features)), index=dates, columns=cols)


def test_zscore_preprocessor_matches_scale_by_train_stats():
    rng = np.random.default_rng(42)
    df = make_synthetic_wide_df(rng)
    n = len(df)
    train, val, test = (
        df.iloc[: int(n * 0.7)],
        df.iloc[int(n * 0.7): int(n * 0.85)],
        df.iloc[int(n * 0.85):],
    )

    train_ref, val_ref, test_ref, mean_ref, std_ref = scale_by_train_stats(train, val, test)

    preprocessor = ZScorePreprocessor()
    preprocessor.fit(train)
    train_wrapped = preprocessor.transform(train)
    val_wrapped = preprocessor.transform(val)
    test_wrapped = preprocessor.transform(test)
    meta = preprocessor.get_metadata()

    # Same deterministic arithmetic re-executed -> tight tolerance, effectively exact.
    np.testing.assert_allclose(train_wrapped.values, train_ref.values, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(val_wrapped.values, val_ref.values, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(test_wrapped.values, test_ref.values, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(meta["train_mean"].values, mean_ref.values, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(meta["train_std"].values, std_ref.values, rtol=1e-12, atol=1e-15)
    assert meta["feature_names"] == train.columns.tolist()
