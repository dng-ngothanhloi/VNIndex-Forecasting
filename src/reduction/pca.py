"""
pca.py – PCAReducer (src/reduction)
========================================
Concrete BaseReducer implementation wrapping sklearn.decomposition.PCA
unchanged (Wrap_Dont_Rewrite, pca-reducer-wrapper Phase 1A).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.decomposition import PCA

from .base import BaseReducer, NotFittedError


class PCAReducer(BaseReducer):
    """Wraps sklearn.decomposition.PCA behind the BaseReducer contract.

    Per Wrap_Dont_Rewrite: fit/transform delegate directly to an
    internally held sklearn.decomposition.PCA instance with no
    post-processing of any numerical value (Requirement 2.8). The
    constructor's n_components and random_state are passed unchanged
    to that internal instance (Requirement 2.3).

    Public API is exactly: fit, transform, fit_transform, get_metadata
    (Requirement 2.2). All other state is private (single-underscore) to
    keep the public surface minimal.
    """

    def __init__(self, n_components: Optional[int] = None, random_state: Optional[int] = None) -> None:
        self._pca = PCA(n_components=n_components, random_state=random_state)
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "PCAReducer":
        self._pca.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError(
                "PCAReducer.transform() called before fit(). Call fit(X) first."
            )
        return self._pca.transform(X)

    def get_metadata(self) -> Dict[str, Any]:
        if not self._is_fitted:
            raise NotFittedError(
                "PCAReducer.get_metadata() called before fit(). Call fit(X) first."
            )
        return {
            "method": "PCA",
            "n_components": self._pca.n_components_,
            "explained_variance_ratio": self._pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(self._pca.explained_variance_ratio_),
            "n_features_in": self._pca.n_features_in_,
            "components_": self._pca.components_,
        }
