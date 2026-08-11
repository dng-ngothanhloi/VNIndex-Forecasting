"""Shape-only tests for the PredictionResult dataclass contract.

Per Master Roadmap Req 6.14/6.16, PredictionResult is DEFINED in this phase
but NOT assembled by any forecaster or orchestration code. These tests only
confirm the dataclass's field structure and immutability -- they do not test
any forecaster constructing one, since no forecaster does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting import PredictionResult


def test_prediction_result_has_exactly_the_five_governed_fields():
    field_names = {f for f in PredictionResult.__dataclass_fields__}
    assert field_names == {"timestamps", "y_true", "y_pred", "model_name", "metadata"}


def test_prediction_result_construction_and_field_access():
    ts = pd.date_range("2024-01-01", periods=3)
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    pr = PredictionResult(
        timestamps=ts,
        y_true=y_true,
        y_pred=y_pred,
        model_name="ARDL",
        metadata={"selected_pair": (5, 2)},
    )
    assert pr.model_name == "ARDL"
    assert pr.metadata == {"selected_pair": (5, 2)}
    assert np.array_equal(pr.y_true, y_true)
    assert np.array_equal(pr.y_pred, y_pred)
    assert (pr.timestamps == ts).all()


def test_prediction_result_is_frozen():
    pr = PredictionResult(
        timestamps=pd.date_range("2024-01-01", periods=1),
        y_true=np.array([1.0]),
        y_pred=np.array([1.0]),
        model_name="ARDL",
        metadata={},
    )
    with pytest.raises(Exception):
        pr.model_name = "LSTM"  # frozen dataclass must reject mutation
