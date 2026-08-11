"""
zscore.py – ZScorePreprocessor (src/preprocessing)
======================================================
Wraps today's scale_by_train_stats arithmetic (helpers/utils.py) behind
the BasePreprocessor contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .base import BasePreprocessor, NotFittedError


class ZScorePreprocessor(BasePreprocessor):
    """Wraps today's scale_by_train_stats arithmetic (helpers/utils.py)
    behind the BasePreprocessor contract.

    No constructor parameters: unlike PCAReducer, Z-score scaling has no
    hyperparameters to configure (Requirement 2 design note).

    Public API is exactly: fit, transform, fit_transform, get_metadata.
    All other state is private (single-underscore) to keep the public
    surface minimal (Requirement 2.2).
    """

    def __init__(self) -> None:
        self._train_mean: Optional[pd.Series] = None
        self._train_std: Optional[pd.Series] = None
        self._feature_names: Optional[List[str]] = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame) -> "ZScorePreprocessor":
        # Identical formula to scale_by_train_stats(train_df, val_df, test_df)
        # in helpers/utils.py: mean() and std() with zero-std replaced by 1.
        self._train_mean = X.mean()
        self._train_std = X.std().replace(0, 1)
        self._feature_names = X.columns.tolist()
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise NotFittedError(
                "ZScorePreprocessor.transform() called before fit(). Call fit(X) first."
            )
        return (X - self._train_mean) / self._train_std

    def get_metadata(self) -> Dict[str, Any]:
        if not self._is_fitted:
            raise NotFittedError(
                "ZScorePreprocessor.get_metadata() called before fit(). Call fit(X) first."
            )
        return {
            "train_mean": self._train_mean,
            "train_std": self._train_std,
            "feature_names": self._feature_names,
        }
