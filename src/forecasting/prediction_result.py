"""
prediction_result.py – PredictionResult dataclass contract (src/forecasting)
=================================================================================
Defines the PredictionResult dataclass contract per the Master Roadmap's
Requirement 6.16: the phase that introduces BaseForecaster must define this
contract's field structure, even though no BaseForecaster implementation in
this phase assembles or returns instances of it.

DEFINED HERE, NOT WIRED: no forecaster constructs a PredictionResult, and no
orchestration code in this phase consumes it (none exists yet). Assembly is
deferred to the future ExperimentPipeline/evaluation boundary (Req 6.14),
which has explicit access to timestamps and y_true as contextual inputs that
BaseForecaster.predict(X) does not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictionResult:
    """Standardized forecast-result contract.

    Fields (per Master Roadmap Req 6.16):
        timestamps: the dates/index corresponding to each prediction
        y_true: the ground-truth values for the same timestamps
        y_pred: the forecaster's predicted values
        model_name: identifies which forecaster produced this result
        metadata: forecaster-specific metadata (e.g. from get_metadata())
    """

    timestamps: pd.Index
    y_true: np.ndarray
    y_pred: np.ndarray
    model_name: str
    metadata: Dict[str, Any]
