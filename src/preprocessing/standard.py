"""
standard.py – StandardPreprocessor (src/preprocessing)
===========================================================
Leakage-safe replacement for the pre-split full-date-range IQR outlier
removal, missing-ratio feature filtering, and backward-fill imputation
previously implemented in src/preprocess_steps/step_3_outliers.py,
step_5_filter_observation_ratio.py, and step_6_fill_and_clean.py
(forecasting-protocol-audit.md: Scientific_Risk=Critical,
Scientific_Correction_Required=Yes for all three -- see Stage 2 of the
pipeline-corrections task).

Governed invariant: every learned statistic (IQR bounds, feature mask,
imputation fallback) is fit on TRAIN ONLY, then applied unchanged to
Val/Test. Imputation is causal: a value at date t may only be filled
using information dated < t (its own split's chronological history, or
the already-processed tail of a strictly preceding split as carry-in
context) -- never backward-filled from a later date.

Design note: unlike ZScorePreprocessor (a single stateless transform
reused identically for train/val/test), this preprocessor's transform
must know whether it is processing the split fit() was called on (pure
self-contained forward-fill) or a LATER split (needs carry-in context
from the immediately preceding split's tail for correct causal
forward-fill across the split boundary). This is exposed as two explicit
methods -- `transform` (self-contained) and `transform_next` (with
carry-in context) -- so the distinction is unambiguous at every call
site (coding-governance: Explicit State Passing), rather than silently
inferred inside one `transform(X)` call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BasePreprocessor, NotFittedError


class StandardPreprocessor(BasePreprocessor):
    """Fits IQR outlier bounds, a missing-ratio feature mask, and
    imputation fallback statistics on Train ONLY; applies the frozen
    state to Train/Val/Test via `transform`/`transform_next`.

    Public API: fit, transform, transform_next, fit_transform, get_metadata.
    """

    def __init__(self, outlier_k: float = 1.5, missing_threshold: float = 0.2) -> None:
        self.outlier_k = outlier_k
        self.missing_threshold = missing_threshold

        self._lower: Optional[pd.Series] = None
        self._upper: Optional[pd.Series] = None
        self._selected_columns: Optional[List[str]] = None
        self._fallback: Optional[pd.Series] = None
        self._is_fitted = False

    # ── fit: learn all state from TRAIN ONLY ──────────────────────────
    def fit(self, X: pd.DataFrame) -> "StandardPreprocessor":
        """Fit on X (the Train split ONLY). Learns, in order:
        1. Per-column IQR outlier bounds (Tukey k) from Train values.
        2. A feature-inclusion mask: columns whose Train missing ratio,
           AFTER masking Train's own outliers, is <= missing_threshold.
        3. A per-column fallback statistic (median) for imputation gaps
           forward-fill cannot resolve, computed on Train's selected
           columns, post outlier-masking.
        Returns self.
        """
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self._lower = q1 - self.outlier_k * iqr
        self._upper = q3 + self.outlier_k * iqr
        self._is_fitted = True  # bounds now available for mask_outliers()

        train_masked = self.mask_outliers(X)
        missing_ratio = train_masked.isnull().mean()
        self._selected_columns = missing_ratio[missing_ratio <= self.missing_threshold].index.tolist()

        self._fallback = train_masked[self._selected_columns].median()
        return self

    def mask_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Set any value outside the Train-fit [lower, upper] bounds to
        NaN, for columns present in both X and the fitted bounds. Uses
        the SAME per-column bounds fit() computed on Train, regardless
        of which split X is -- Val/Test never contribute their own
        bounds (Requirement 2.2 of Stage 2). Public: also used by
        diagnostics/reporting code that wants to visualize the effect of
        the Train-fit bounds on arbitrary (e.g. full-range) data without
        that visualization use ever influencing the fitted state itself."""
        if not self._is_fitted:
            raise NotFittedError(
                "StandardPreprocessor.mask_outliers() called before fit(). Call fit(X) first."
            )
        cols = [c for c in X.columns if c in self._lower.index]
        out = X.copy()
        for c in cols:
            mask = (out[c] < self._lower[c]) | (out[c] > self._upper[c])
            out.loc[mask, c] = np.nan
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a SELF-CONTAINED split (typically Train, the same
        split fit() was called on): mask outliers using the Train-fit
        bounds, select the Train-fit feature columns, then forward-fill
        chronologically within X itself. Any leading NaN forward-fill
        cannot resolve falls back to the Train-fit median -- never
        backward-fill from a later date within X."""
        if not self._is_fitted:
            raise NotFittedError(
                "StandardPreprocessor.transform() called before fit(). Call fit(X) first."
            )
        masked = self.mask_outliers(X)[self._selected_columns]
        filled = masked.ffill()
        filled = filled.fillna(self._fallback)
        return filled

    def transform_next(self, X: pd.DataFrame, context_tail: pd.DataFrame) -> pd.DataFrame:
        """Transform a split that chronologically FOLLOWS an
        already-processed split (Val following Train, or Test following
        Train+Val): mask outliers + select columns using the SAME
        Train-fit state, then forward-fill with `context_tail` (the
        tail of the already-processed preceding split) providing
        carry-in history, so the first row(s) of X are never stranded
        with no prior observation.

        `context_tail` MUST already be fully processed (outlier-masked,
        column-selected, imputed) and MUST end strictly before X begins
        -- this function never reads from X's own future rows and never
        reads from context_tail's successor.
        """
        if not self._is_fitted:
            raise NotFittedError(
                "StandardPreprocessor.transform_next() called before fit(). Call fit(X) first."
            )
        if not context_tail.index.max() < X.index.min():
            raise ValueError(
                "context_tail must end strictly before X begins (chronological, non-overlapping carry-in)."
            )
        masked = self.mask_outliers(X)[self._selected_columns]
        combined = pd.concat([context_tail[self._selected_columns], masked], axis=0)
        filled = combined.ffill()
        filled = filled.fillna(self._fallback)
        return filled.loc[X.index]

    def get_metadata(self) -> Dict[str, Any]:
        if not self._is_fitted:
            raise NotFittedError(
                "StandardPreprocessor.get_metadata() called before fit(). Call fit(X) first."
            )
        return {
            "outlier_k": self.outlier_k,
            "missing_threshold": self.missing_threshold,
            "lower_bounds": self._lower,
            "upper_bounds": self._upper,
            "selected_columns": self._selected_columns,
            "n_selected_columns": len(self._selected_columns),
            "fallback": self._fallback,
        }
