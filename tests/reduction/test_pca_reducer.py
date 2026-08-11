"""
test_pca_reducer.py – PCAReducer unit + property + Synthetic_Parity_Fixture
================================================================================
Feature: pca-reducer-wrapper

Covers:
- Task 4.2: unit test for PCAReducer public API surface
- Task 4.3: Property 3 (public surface never grows)
- Task 4.4: Property 4 (constructor arguments pass through unchanged)
- Task 4.5: Property 5 (NotFittedError before fit)
- Task 4.6: Property 7 (get_metadata returns the complete, correct
  Metadata_Contract)
- Task 6.2: unit test for package re-exports
- Task 7.1: Synthetic_Parity_Fixture (mandatory Parity_Gate regression
  test)
- Task 7.2: Property 6 (numerical equivalence with direct
  sklearn.decomposition.PCA usage)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reduction.pca import PCAReducer  # noqa: E402
from reduction.base import NotFittedError  # noqa: E402

_EXPECTED_PUBLIC_API = {"fit", "transform", "fit_transform", "get_metadata"}


def _public_attrs(obj) -> set:
    return {name for name in dir(obj) if not name.startswith("_")}


# ─────────────────────────────────────────────────────────────────
# Task 6.2: unit test - package re-exports
# ─────────────────────────────────────────────────────────────────
def test_package_reexports_succeed():
    from reduction import BaseReducer, NotFittedError as ReExportedNotFittedError, PCAReducer as ReExportedPCAReducer  # noqa: F401


# ─────────────────────────────────────────────────────────────────
# Task 4.2: unit test - public API surface is exactly the 4 methods
# ─────────────────────────────────────────────────────────────────
def test_pca_reducer_public_api_is_minimal():
    r = PCAReducer()
    assert _public_attrs(r) == _EXPECTED_PUBLIC_API


# ─────────────────────────────────────────────────────────────────
# Task 4.3 / Property 3: PCAReducer's public surface never grows
# **Property 3: PCAReducer's public surface never grows**
# **Validates: Requirements 2.2**
# ─────────────────────────────────────────────────────────────────
@given(
    n_obs=st.integers(min_value=5, max_value=50),
    n_features=st.integers(min_value=2, max_value=10),
    n_components=st.integers(min_value=1, max_value=2),
    random_state=st.one_of(st.none(), st.integers(min_value=0, max_value=1000)),
)
@settings(max_examples=100)
def test_pca_reducer_public_surface_never_grows(n_obs, n_features, n_components, random_state):
    n_components = min(n_components, n_features)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_obs, n_features))

    r = PCAReducer(n_components=n_components, random_state=random_state)
    assert _public_attrs(r) == _EXPECTED_PUBLIC_API

    r.fit(X)
    assert _public_attrs(r) == _EXPECTED_PUBLIC_API

    r.transform(X)
    assert _public_attrs(r) == _EXPECTED_PUBLIC_API


# ─────────────────────────────────────────────────────────────────
# Task 4.4 / Property 4: Constructor arguments pass through unchanged
# **Property 4: Constructor arguments pass through unchanged**
# **Validates: Requirements 2.3**
# ─────────────────────────────────────────────────────────────────
@given(
    n_components=st.one_of(
        st.none(),
        st.integers(min_value=1, max_value=10),
        st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
    ),
    random_state=st.one_of(st.none(), st.integers(min_value=0, max_value=10_000)),
)
@settings(max_examples=100)
def test_constructor_arguments_pass_through_unchanged(n_components, random_state):
    r = PCAReducer(n_components=n_components, random_state=random_state)
    assert r._pca.n_components == n_components
    assert r._pca.random_state == random_state


# ─────────────────────────────────────────────────────────────────
# Task 4.5 / Property 5: Methods requiring fit raise NotFittedError
# **Property 5: Methods requiring fit raise NotFittedError before fit is called**
# **Validates: Requirements 2.6, 2.7**
# ─────────────────────────────────────────────────────────────────
@given(
    n_obs=st.integers(min_value=1, max_value=20),
    n_features=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=100)
def test_unfitted_methods_raise_not_fitted_error(n_obs, n_features):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n_obs, n_features))

    r = PCAReducer()

    with pytest.raises(NotFittedError):
        r.transform(X)

    with pytest.raises(NotFittedError):
        r.get_metadata()


# ─────────────────────────────────────────────────────────────────
# Task 4.6 / Property 7: get_metadata returns the complete, correct
# Metadata_Contract
# **Property 7: get_metadata returns the complete, correct Metadata_Contract**
# **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**
# ─────────────────────────────────────────────────────────────────
@given(
    n_obs=st.integers(min_value=5, max_value=50),
    n_features=st.integers(min_value=2, max_value=10),
    n_components=st.integers(min_value=1, max_value=2),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=100)
def test_get_metadata_returns_complete_metadata_contract(n_obs, n_features, n_components, seed):
    n_components = min(n_components, n_features, n_obs)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_obs, n_features))

    r = PCAReducer(n_components=n_components, random_state=0)
    r.fit(X)
    meta = r.get_metadata()

    required_keys = {
        "method",
        "n_components",
        "explained_variance_ratio",
        "cumulative_explained_variance",
        "n_features_in",
        "components_",
    }
    assert required_keys.issubset(meta.keys())
    assert meta["method"] == "PCA"
    assert meta["n_components"] == r._pca.n_components_
    np.testing.assert_array_equal(meta["explained_variance_ratio"], r._pca.explained_variance_ratio_)
    np.testing.assert_array_equal(
        meta["cumulative_explained_variance"], np.cumsum(r._pca.explained_variance_ratio_)
    )
    assert meta["n_features_in"] == r._pca.n_features_in_
    np.testing.assert_array_equal(meta["components_"], r._pca.components_)


# ─────────────────────────────────────────────────────────────────
# Task 7.1: Synthetic_Parity_Fixture (mandatory Parity_Gate, non-optional)
# ─────────────────────────────────────────────────────────────────
def test_synthetic_parity_fixture_matches_direct_sklearn_pca():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 15))

    legacy_pca = PCA(n_components=2, random_state=0)
    legacy_pca.fit(X)
    legacy_transform = legacy_pca.transform(X)

    reducer = PCAReducer(n_components=2, random_state=0)
    reducer.fit(X)
    wrapped_transform = reducer.transform(X)
    meta = reducer.get_metadata()

    np.testing.assert_allclose(wrapped_transform, legacy_transform, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        meta["explained_variance_ratio"], legacy_pca.explained_variance_ratio_, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(meta["components_"], legacy_pca.components_, rtol=1e-10, atol=1e-12)


# ─────────────────────────────────────────────────────────────────
# Task 7.2 / Property 6: PCAReducer is numerically equivalent to direct
# sklearn.decomposition.PCA usage
# **Property 6: PCAReducer is numerically equivalent to direct sklearn.decomposition.PCA usage**
# **Validates: Requirements 2.4, 2.5, 2.8, 4.4, 4.5, 6.2, 6.3, 6.4, 6.5**
# ─────────────────────────────────────────────────────────────────
@given(
    n_obs=st.integers(min_value=5, max_value=60),
    n_features=st.integers(min_value=2, max_value=15),
    n_components=st.integers(min_value=1, max_value=2),
    random_state=st.one_of(st.none(), st.integers(min_value=0, max_value=1000)),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=100)
def test_pca_reducer_numerically_equivalent_to_direct_sklearn_pca(
    n_obs, n_features, n_components, random_state, seed
):
    n_components = min(n_components, n_features, n_obs)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_obs, n_features))

    legacy_pca = PCA(n_components=n_components, random_state=random_state)
    legacy_pca.fit(X)
    legacy_transform = legacy_pca.transform(X)

    reducer = PCAReducer(n_components=n_components, random_state=random_state)
    reducer.fit(X)
    wrapped_transform = reducer.transform(X)
    meta = reducer.get_metadata()

    np.testing.assert_allclose(wrapped_transform, legacy_transform, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        meta["explained_variance_ratio"], legacy_pca.explained_variance_ratio_, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(meta["components_"], legacy_pca.components_, rtol=1e-10, atol=1e-12)
