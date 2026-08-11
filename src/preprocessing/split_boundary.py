"""
split_boundary.py – Split_Boundary data contract (src/preprocessing)
========================================================================
Normalizes the output of today's two differently-shaped split functions
(helpers.utils.split_by_time -> DataSplits, helpers.split_utils.
split_time_series_overlap_val -> SplitResult) into one consistent,
auditable shape carrying the three DataFrames and their date-range
boundaries as first-class fields.

This is the ONLY file in src/preprocessing/ that touches split logic. It
does not change which underlying function is called, its arguments, or
its internal splitting behavior for either branch (Requirement 5.5).
Deliberately a plain frozen dataclass, not a new abstract base class
(Requirement 5.1, Requirement 9.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from src.utils import split_by_time as _split_by_time_nonoverlap
from src.preprocessing.split_utils import split_time_series_overlap_val as _split_time_series_overlap_val


@dataclass(frozen=True)
class Split_Boundary:
    """Explicit, auditable train/val/test data contract.

    Normalizes helpers.utils.DataSplits and helpers.split_utils.SplitResult
    into one consistent shape carrying the three DataFrames AND their date
    boundaries as first-class fields (neither source dataclass exposes date
    boundaries directly — both require calling .index.min()/.max() at each
    use site today).
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_start: Any
    train_end: Any
    val_start: Any
    val_end: Any
    test_start: Any
    test_end: Any


def build_split_boundary(df_pivot: pd.DataFrame, prep_cfg: Dict[str, Any]) -> Split_Boundary:
    """Construct a Split_Boundary from df_pivot per prep_cfg["use_overlap_val"].

    This is the ONLY new code that touches split logic. It does not change
    which underlying function is called, its arguments, or its internal
    splitting behavior for either branch (Requirement 5.5) — it strictly
    calls the same two functions src/preprocess.py calls today, with the
    same arguments, and repackages their output.
    """
    use_overlap = prep_cfg.get("use_overlap_val", False)

    if use_overlap:
        # Same call as today's src/preprocess.py overlap branch — never
        # helpers.split_utils's own split_by_time dispatcher (Requirement 5.6/9.7).
        result = _split_time_series_overlap_val(
            df_pivot,
            train_ratio=float(prep_cfg.get("train_ratio", 0.7)),
            val_overlap_ratio=float(prep_cfg.get("val_overlap_ratio", 0.20)),
        )
    else:
        # Same call as today's src/preprocess.py non-overlap branch —
        # helpers.utils.split_by_time, not helpers.split_utils's split_by_time.
        result = _split_by_time_nonoverlap(
            df_pivot,
            train_ratio=float(prep_cfg.get("train_ratio", 0.7)),
            val_ratio=float(prep_cfg.get("val_ratio", 0.15)),
        )

    train, val, test = result.train, result.val, result.test
    return Split_Boundary(
        train=train,
        val=val,
        test=test,
        train_start=train.index.min(),
        train_end=train.index.max(),
        val_start=val.index.min(),
        val_end=val.index.max(),
        test_start=test.index.min(),
        test_end=test.index.max(),
    )
