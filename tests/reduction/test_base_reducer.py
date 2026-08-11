"""
test_base_reducer.py – BaseReducer contract tests
======================================================
Feature: pca-reducer-wrapper

Covers:
- Task 2.2: unit test for BaseReducer non-instantiability and
  NotFittedError type
- Task 2.3: Property 2 (incomplete BaseReducer subclasses cannot be
  instantiated)
- Task 2.4: Property 1 (fit_transform composes fit then transform)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import sklearn.exceptions
from hypothesis import given, settings, strategies as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reduction.base import BaseReducer, NotFittedError  # noqa: E402


# ─────────────────────────────────────────────────────────────────
# Task 2.2: unit test - BaseReducer cannot be instantiated directly,
# and NotFittedError is a plain RuntimeError subclass, not sklearn's.
# ─────────────────────────────────────────────────────────────────
def test_base_reducer_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseReducer()


def test_not_fitted_error_is_runtime_error_not_sklearn():
    assert issubclass(NotFittedError, RuntimeError) is True
    assert issubclass(NotFittedError, sklearn.exceptions.NotFittedError) is False


# ─────────────────────────────────────────────────────────────────
# Minimal concrete subclass used by Property 1's test.
# ─────────────────────────────────────────────────────────────────
class _MinimalReducer(BaseReducer):
    """Trivial concrete BaseReducer: stores X on fit, adds 1 on transform."""

    def __init__(self) -> None:
        self._fitted_value = None
        self.fit_calls = 0
        self.transform_calls = 0

    def fit(self, X):
        self.fit_calls += 1
        self._fitted_value = X
        return self

    def transform(self, X):
        self.transform_calls += 1
        return X + 1

    def get_metadata(self):
        return {"fitted_value": self._fitted_value}


# ─────────────────────────────────────────────────────────────────
# Task 2.4 / Property 1: fit_transform composes fit then transform
# **Property 1: fit_transform composes fit then transform**
# **Validates: Requirements 1.5**
# ─────────────────────────────────────────────────────────────────
@given(
    values=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=100)
def test_fit_transform_composes_fit_then_transform(values):
    X = np.array(values)

    composed = _MinimalReducer()
    composed_result = composed.fit_transform(X)

    separate = _MinimalReducer()
    separate.fit(X)
    separate_result = separate.transform(X)

    np.testing.assert_array_equal(composed_result, separate_result)
    assert composed.fit_calls == 1
    assert composed.transform_calls == 1
    np.testing.assert_array_equal(
        composed.get_metadata()["fitted_value"], separate.get_metadata()["fitted_value"]
    )


# ─────────────────────────────────────────────────────────────────
# Task 2.3 / Property 2: Incomplete BaseReducer subclasses cannot be
# instantiated
# **Property 2: Incomplete BaseReducer subclasses cannot be instantiated**
# **Validates: Requirements 1.6, 2.1**
# ─────────────────────────────────────────────────────────────────
def _fit_impl(self, X):
    return self


def _transform_impl(self, X):
    return X


def _get_metadata_impl(self):
    return {}


_METHOD_IMPLS = {
    "fit": _fit_impl,
    "transform": _transform_impl,
    "get_metadata": _get_metadata_impl,
}


@given(omitted_method=st.sampled_from(["fit", "transform", "get_metadata"]))
@settings(max_examples=100)
def test_incomplete_subclass_raises_type_error(omitted_method):
    namespace = {
        name: impl for name, impl in _METHOD_IMPLS.items() if name != omitted_method
    }
    IncompleteSubclass = type("IncompleteSubclass", (BaseReducer,), namespace)

    with pytest.raises(TypeError):
        IncompleteSubclass()


def test_complete_subclass_instantiates_successfully():
    CompleteSubclass = type("CompleteSubclass", (BaseReducer,), dict(_METHOD_IMPLS))
    instance = CompleteSubclass()
    assert isinstance(instance, BaseReducer)
