"""
noreduction.py – NoReduction identity transform (src/reduction)
================================================================
Concrete BaseReducer implementation: fit records feature dimensions,
transform returns input unchanged. Used for raw-feature baselines
(NoReduction -> ARDL, NoReduction -> LSTM) through the same experiment
path as PCAReducer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseReducer, NotFittedError


class NoReduction(BaseReducer):
    """Identity-transform reducer: fit records feature count/names,
    transform returns X unchanged.

    Supports both np.ndarray and pd.DataFrame inputs. When X is a
    DataFrame, column names are preserved in metadata; for plain ndarray,
    feature_names is None.

    Public API: fit, transform, fit_transform (inherited), get_metadata.
    """

    def __init__(self) -> None:
        self._n_features_in: Optional[int] = None
        self._feature_names: Optional[List[str]] = None
        self._is_fitted: bool = False

    def fit(self, X) -> "NoReduction":
        """Record feature dimensions. No computation on X values."""
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            self._n_features_in = X.shape[1]
        else:
            X = np.asarray(X)
            self._n_features_in = X.shape[1] if X.ndim >= 2 else 1
            self._feature_names = None
        self._is_fitted = True
        return self

    def transform(self, X):
        """Return X unchanged. Validates fitted state and feature dimension."""
        if not self._is_fitted:
            raise NotFittedError(
                "NoReduction.transform() called before fit(). Call fit(X) first."
            )

        if isinstance(X, pd.DataFrame):
            n_features = X.shape[1]
        else:
            X_arr = np.asarray(X)
            n_features = X_arr.shape[1] if X_arr.ndim >= 2 else 1

        if n_features != self._n_features_in:
            raise ValueError(
                f"NoReduction.transform() received input with {n_features} "
                f"features, but fit() recorded {self._n_features_in} features."
            )

        # Identity: return X unchanged, no copy, no mutation.
        return X

    def get_metadata(self) -> Dict[str, Any]:
        """Return minimal identity-transform metadata."""
        if not self._is_fitted:
            raise NotFittedError(
                "NoReduction.get_metadata() called before fit(). Call fit(X) first."
            )
        return {
            "method": "none",
            "n_features_in": self._n_features_in,
            "n_features_out": self._n_features_in,
            "compression_ratio": 1.0,
            "dim_reduction_pct": 0.0,
            "feature_names": self._feature_names,
            "fitted": True,
        }
