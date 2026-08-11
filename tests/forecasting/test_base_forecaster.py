"""Unit + property tests for BaseForecaster's abstract contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st

from src.forecasting.base import BaseForecaster, NotFittedError


def test_base_forecaster_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseForecaster()


class _MinimalForecaster(BaseForecaster):
    """Minimal concrete subclass for testing the abstract contract itself."""

    def __init__(self):
        self._fitted = False
        self._offset = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._offset = float(np.mean(y_train))
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise NotFittedError("not fitted")
        return np.full(len(X), self._offset)

    def get_metadata(self):
        if not self._fitted:
            raise NotFittedError("not fitted")
        return {"offset": self._offset}


def test_minimal_concrete_subclass_instantiates_successfully():
    f = _MinimalForecaster()
    assert f is not None


# Feature: phase-3b-r-ardl-forecaster-plan, Property: Incomplete BaseForecaster
# subclasses cannot be instantiated
@given(missing=st.sampled_from(["fit", "predict", "get_metadata"]))
def test_incomplete_subclass_cannot_be_instantiated(missing):
    methods = {
        "fit": lambda self, X_train, y_train, X_val=None, y_val=None: self,
        "predict": lambda self, X: np.zeros(len(X)),
        "get_metadata": lambda self: {},
    }
    del methods[missing]
    Incomplete = type("Incomplete", (BaseForecaster,), methods)
    with pytest.raises(TypeError):
        Incomplete()


def test_fit_returns_self_and_predict_uses_fitted_state():
    f = _MinimalForecaster()
    result = f.fit(pd.DataFrame({"x": [1, 2, 3]}), pd.Series([10.0, 20.0, 30.0]))
    assert result is f
    preds = f.predict(pd.DataFrame({"x": [1, 2]}))
    assert np.allclose(preds, 20.0)


def test_predict_raises_not_fitted_error_before_fit():
    f = _MinimalForecaster()
    with pytest.raises(NotFittedError):
        f.predict(pd.DataFrame({"x": [1]}))


def test_get_metadata_raises_not_fitted_error_before_fit():
    f = _MinimalForecaster()
    with pytest.raises(NotFittedError):
        f.get_metadata()
