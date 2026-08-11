"""
base.py – BaseForecaster abstract contract (src/forecasting)
================================================================
Governed contract for forecasting methods, mirroring the same
ABC + custom-NotFittedError pattern established by src/reduction (Phase 1A)
and src/preprocessing (Phase 3A).

Frozen contract (per Master Roadmap Req 6.4-6.6, 6.13):
    fit(X_train, y_train, X_val=None, y_val=None) -> "BaseForecaster"
    predict(X) -> np.ndarray                        # y_pred ONLY, never PredictionResult
    get_metadata() -> dict

No fit_transform: unlike BaseReducer/BasePreprocessor, forecasters do not
transform their input; they fit on train(+val) and predict on a distinct
held-out X. X_val/y_val are optional so forecasters that don't use a
validation split during fitting are not forced to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class NotFittedError(RuntimeError):
    """Raised when a BaseForecaster method that requires a prior fit(...)
    is called before fit(...) has been called successfully.

    A plain RuntimeError subclass defined in this package, mirroring the
    src/reduction and src/preprocessing NotFittedError precedent, so
    BaseForecaster's contract does not depend on any specific backing
    library's exception hierarchy.
    """


class BaseForecaster(ABC):
    """Governed abstract contract for forecasting methods.

    Subclasses MUST implement fit, predict, and get_metadata. Instantiating
    a subclass that leaves any of these abstract raises TypeError (enforced
    by abc.ABC + @abstractmethod).
    """

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BaseForecaster":
        """Fit the forecaster on X_train/y_train, optionally using X_val/y_val
        (e.g. for out-of-sample model selection). Returns self."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return y_pred for X using state stored during fit. Returns ONLY
        y_pred (Req 6.13) — never a PredictionResult; timestamps and y_true
        are not available as inputs to predict(X) and are assembled later,
        at the ExperimentPipeline/evaluation boundary (Req 6.14)."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return the fitted forecaster's metadata contract as a dict."""
        raise NotImplementedError
