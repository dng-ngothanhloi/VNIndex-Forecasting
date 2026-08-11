"""
test_noreduction.py – Unit tests for NoReduction identity transform
====================================================================
Covers: fit, transform, fit_transform, get_metadata, error handling,
ndarray/DataFrame support, feature-dimension mismatch, no-mutation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reduction.noreduction import NoReduction
from src.reduction.base import NotFittedError


# ─────────────────────────────────────────────────────────────────
# 1. fit stores feature count
# ─────────────────────────────────────────────────────────────────
def test_fit_stores_feature_count():
    nr = NoReduction()
    X = np.random.randn(10, 5)
    nr.fit(X)
    meta = nr.get_metadata()
    assert meta["n_features_in"] == 5
    assert meta["n_features_out"] == 5


# ─────────────────────────────────────────────────────────────────
# 2. transform before fit raises NotFittedError
# ─────────────────────────────────────────────────────────────────
def test_transform_before_fit_raises_not_fitted_error():
    nr = NoReduction()
    X = np.random.randn(10, 5)
    with pytest.raises(NotFittedError):
        nr.transform(X)


def test_get_metadata_before_fit_raises_not_fitted_error():
    nr = NoReduction()
    with pytest.raises(NotFittedError):
        nr.get_metadata()


# ─────────────────────────────────────────────────────────────────
# 3. transform preserves shape
# ─────────────────────────────────────────────────────────────────
def test_transform_preserves_shape_ndarray():
    nr = NoReduction()
    X = np.random.randn(20, 7)
    nr.fit(X)
    result = nr.transform(X)
    assert result.shape == (20, 7)


def test_transform_preserves_shape_dataframe():
    nr = NoReduction()
    X = pd.DataFrame(np.random.randn(15, 4), columns=["A", "B", "C", "D"])
    nr.fit(X)
    result = nr.transform(X)
    assert result.shape == (15, 4)


# ─────────────────────────────────────────────────────────────────
# 4. transform preserves numerical values
# ─────────────────────────────────────────────────────────────────
def test_transform_preserves_values_ndarray():
    nr = NoReduction()
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    nr.fit(X)
    result = nr.transform(X)
    np.testing.assert_array_equal(result, X)


def test_transform_preserves_values_dataframe():
    nr = NoReduction()
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    nr.fit(X)
    result = nr.transform(X)
    pd.testing.assert_frame_equal(result, X)


# ─────────────────────────────────────────────────────────────────
# 5. fit_transform is identity
# ─────────────────────────────────────────────────────────────────
def test_fit_transform_is_identity_ndarray():
    nr = NoReduction()
    X = np.random.randn(10, 3)
    result = nr.fit_transform(X)
    np.testing.assert_array_equal(result, X)
    assert nr._is_fitted is True


def test_fit_transform_is_identity_dataframe():
    nr = NoReduction()
    X = pd.DataFrame(np.random.randn(8, 5), columns=[f"f{i}" for i in range(5)])
    result = nr.fit_transform(X)
    pd.testing.assert_frame_equal(result, X)


# ─────────────────────────────────────────────────────────────────
# 6. ndarray supported
# ─────────────────────────────────────────────────────────────────
def test_ndarray_roundtrip():
    nr = NoReduction()
    X = np.arange(30).reshape(10, 3).astype(float)
    nr.fit(X)
    result = nr.transform(X)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, X)


# ─────────────────────────────────────────────────────────────────
# 7. DataFrame supported
# ─────────────────────────────────────────────────────────────────
def test_dataframe_roundtrip():
    nr = NoReduction()
    X = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [4.0, 5.0, 6.0]})
    nr.fit(X)
    result = nr.transform(X)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["col1", "col2"]


# ─────────────────────────────────────────────────────────────────
# 8. optional column metadata preserved
# ─────────────────────────────────────────────────────────────────
def test_dataframe_metadata_preserved():
    nr = NoReduction()
    cols = ["VNM", "FPT", "VIC"]
    X = pd.DataFrame(np.random.randn(5, 3), columns=cols)
    nr.fit(X)
    meta = nr.get_metadata()
    assert meta["feature_names"] == cols
    assert meta["method"] == "none"
    assert meta["n_features_in"] == 3
    assert meta["n_features_out"] == 3
    assert meta["dim_reduction_pct"] == 0.0
    assert meta["compression_ratio"] == 1.0


def test_ndarray_metadata_feature_names_is_none():
    nr = NoReduction()
    X = np.random.randn(5, 4)
    nr.fit(X)
    meta = nr.get_metadata()
    assert meta["feature_names"] is None
    assert meta["n_features_in"] == 4


# ─────────────────────────────────────────────────────────────────
# 9. feature-dimension mismatch raises ValueError
# ─────────────────────────────────────────────────────────────────
def test_feature_dimension_mismatch_raises():
    nr = NoReduction()
    X_fit = np.random.randn(10, 5)
    X_bad = np.random.randn(10, 3)
    nr.fit(X_fit)
    with pytest.raises(ValueError, match="5 features"):
        nr.transform(X_bad)


def test_feature_dimension_mismatch_dataframe():
    nr = NoReduction()
    X_fit = pd.DataFrame(np.random.randn(10, 4), columns=["A", "B", "C", "D"])
    X_bad = pd.DataFrame(np.random.randn(10, 2), columns=["A", "B"])
    nr.fit(X_fit)
    with pytest.raises(ValueError):
        nr.transform(X_bad)


# ─────────────────────────────────────────────────────────────────
# 10. returned output does not unexpectedly mutate caller input
# ─────────────────────────────────────────────────────────────────
def test_transform_does_not_mutate_input_ndarray():
    nr = NoReduction()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_copy = X.copy()
    nr.fit(X)
    result = nr.transform(X)
    # Modifying result should not change the original if identity
    # (NoReduction returns X itself — but no MUTATION occurs inside transform)
    np.testing.assert_array_equal(X, X_copy)


def test_transform_does_not_mutate_input_dataframe():
    nr = NoReduction()
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    X_copy = X.copy()
    nr.fit(X)
    nr.transform(X)
    pd.testing.assert_frame_equal(X, X_copy)


# ─────────────────────────────────────────────────────────────────
# Integration-adjacent: NoReduction outputs correct metadata shape
# ─────────────────────────────────────────────────────────────────
def test_metadata_keys_complete():
    nr = NoReduction()
    X = np.random.randn(10, 318)
    nr.fit(X)
    meta = nr.get_metadata()
    expected_keys = {"method", "n_features_in", "n_features_out",
                     "compression_ratio", "dim_reduction_pct",
                     "feature_names", "fitted"}
    assert set(meta.keys()) == expected_keys


def test_fit_318_features():
    """Simulate NoReduction with real feature count (318 stocks)."""
    nr = NoReduction()
    X = np.random.randn(536, 318)
    nr.fit(X)
    meta = nr.get_metadata()
    assert meta["n_features_in"] == 318
    assert meta["n_features_out"] == 318
    assert meta["dim_reduction_pct"] == 0.0
    result = nr.transform(X)
    assert result.shape == (536, 318)
    np.testing.assert_array_equal(result, X)
