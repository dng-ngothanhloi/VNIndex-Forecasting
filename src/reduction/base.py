"""
base.py – BaseReducer abstract contract (src/reduction)
===========================================================
Governed abstract contract for dimensionality-reduction methods
(pca-reducer-wrapper Phase 1A). Not gated on any other spec — this is
the first real implementation of BaseReducer/NotFittedError in this
codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class NotFittedError(RuntimeError):
    """Raised when a BaseReducer method that requires a prior fit(X)
    is called before fit(X) has been called successfully.

    A plain RuntimeError subclass defined in this package — deliberately
    NOT sklearn.exceptions.NotFittedError or any other third-party
    exception type — so that BaseReducer's contract does not couple to
    any specific backing library (Requirement 1.7, Requirement 1.8). Any
    future BaseReducer implementation (KPCAReducer, KTPCAReducer,
    NoReduction) reuses this same type rather than defining its own.
    """


class BaseReducer(ABC):
    """Governed abstract contract for dimensionality-reduction methods.

    Subclasses MUST implement fit, transform, and get_metadata.
    Instantiating a subclass that leaves any of these abstract raises
    TypeError (enforced by abc.ABC + @abstractmethod), satisfying
    Requirement 1.6.
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseReducer":
        """Fit the reducer on X. Returns self."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using state stored during fit."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return the fitted reducer's metadata contract as a dict."""
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Concrete convenience method: fit(X) then transform(X)."""
        self.fit(X)
        return self.transform(X)
