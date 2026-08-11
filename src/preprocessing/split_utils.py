"""
split_utils.py – Utility functions for train/val/test splitting strategies
Supports both non-overlap and overlap validation approaches
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass
class SplitResult:
    """Container for split results"""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    val_is_overlap: bool = False
    metadata: dict = None


def split_time_series_nonoverlap(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> SplitResult:
    """
    Traditional non-overlapping split.
    
    Args:
        df: DataFrame with datetime index
        train_ratio: fraction for train
        val_ratio: fraction for validation
        
    Returns:
        SplitResult with train, val, test
    """
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    
    metadata = {
        'strategy': 'non_overlap',
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': 1 - train_ratio - val_ratio,
        'n_train': len(train),
        'n_val': len(val),
        'n_test': len(test),
        'train_period': (train.index.min(), train.index.max()),
        'val_period': (val.index.min(), val.index.max()),
        'test_period': (test.index.min(), test.index.max()),
    }
    
    return SplitResult(
        train=train,
        val=val,
        test=test,
        val_is_overlap=False,
        metadata=metadata,
    )


def split_time_series_overlap_val(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_overlap_ratio: float = 0.20,
) -> SplitResult:
    """
    Overlapping validation strategy.
    
    Validation = last (val_overlap_ratio × train) obs of train.
    Test = remaining data after train.
    
    Example:
        total=1000, train_ratio=0.70 → train=700
        val_overlap_ratio=0.20 → val = train[560:700] (last 140 of train)
        test = obs[700:1000]
        
    Args:
        df: DataFrame with datetime index
        train_ratio: fraction for train (e.g. 0.70)
        val_overlap_ratio: fraction of train for validation (e.g. 0.20)
        
    Returns:
        SplitResult with overlapping val
    """
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    
    # Val = last (val_overlap_ratio × train) rows of train
    n_val_overlap = int(n_train * val_overlap_ratio)
    val_start_idx = n_train - n_val_overlap
    
    train = df.iloc[:n_train].copy()
    val = df.iloc[val_start_idx:n_train].copy()
    test = df.iloc[n_train:].copy()
    
    metadata = {
        'strategy': 'overlap_validation',
        'train_ratio': train_ratio,
        'val_overlap_ratio': val_overlap_ratio,
        'test_ratio': 1 - train_ratio,
        'n_train': len(train),
        'n_val': len(val),
        'n_test': len(test),
        'val_overlap_size': n_val_overlap,
        'train_period': (train.index.min(), train.index.max()),
        'val_period': (val.index.min(), val.index.max()),
        'test_period': (test.index.min(), test.index.max()),
    }
    
    return SplitResult(
        train=train,
        val=val,
        test=test,
        val_is_overlap=True,
        metadata=metadata,
    )


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    use_overlap_val: bool = False,
) -> SplitResult:
    """
    Unified time-series split function.
    
    Args:
        df: DataFrame with datetime index
        train_ratio: fraction for train
        val_ratio: fraction for val (ignored if use_overlap_val=True)
        use_overlap_val: If True, use overlap strategy with val_ratio as overlap_ratio
        
    Returns:
        SplitResult
    """
    if use_overlap_val:
        return split_time_series_overlap_val(
            df,
            train_ratio=train_ratio,
            val_overlap_ratio=val_ratio,  # reinterpret as overlap ratio
        )
    else:
        return split_time_series_nonoverlap(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )


def print_split_summary(result: SplitResult) -> None:
    """Print formatted summary of split results."""
    print("=" * 70)
    print("DATA SPLIT SUMMARY")
    print("=" * 70)
    print(f"Strategy: {result.metadata['strategy']}")
    print()
    print(f"Train:  {result.metadata['n_train']:4d} obs  "
          f"[{result.metadata['train_period'][0].date()} → "
          f"{result.metadata['train_period'][1].date()}]")
    print(f"Val:    {result.metadata['n_val']:4d} obs  "
          f"[{result.metadata['val_period'][0].date()} → "
          f"{result.metadata['val_period'][1].date()}]"
          f"  {'(overlap)' if result.val_is_overlap else '(separate)'}")
    print(f"Test:   {result.metadata['n_test']:4d} obs  "
          f"[{result.metadata['test_period'][0].date()} → "
          f"{result.metadata['test_period'][1].date()}]")
    print()
    print(f"Ratios: Train={result.metadata['train_ratio']:.0%}, "
          f"Val={result.metadata.get('val_ratio', result.metadata.get('val_overlap_ratio', 0)):.0%}, "
          f"Test={result.metadata['test_ratio']:.0%}")
    
    if result.val_is_overlap:
        print(f"\nValidation overlap: {result.metadata['val_overlap_size']} obs "
              f"from end of train")
    
    print("=" * 70)
