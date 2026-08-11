"""
utils.py  –  Shared helpers for VN-Index pipeline (canonical, src/utils.py)
=============================================================================
Import: from src.utils import ...

Migrated from helpers/utils.py during the architecture-convergence task
(generic reusable helpers -> src/utils.py; split helpers moved separately
to src/preprocessing/split_utils.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────
# NUMBER PARSING
# ─────────────────────────────────────────────────────────────────
def clean_number(value: object) -> float:
    """Convert strings like '38,200.0' / '1.2K' / '3M' / '1B' to float."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    if not text:
        return np.nan
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9}
    suffix = text[-1].upper()
    if suffix in multipliers:
        try:
            return float(text[:-1]) * multipliers[suffix]
        except ValueError:
            return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_percent_to_float(series: pd.Series) -> pd.Series:
    """Convert '1.52%' → 1.52 (float)."""
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False), errors="coerce"
    )


def safe_make_columns_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Apply clean_number to existing columns only."""
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = result[col].apply(clean_number)
    return result


# ─────────────────────────────────────────────────────────────────
# OUTLIER REMOVAL
# ─────────────────────────────────────────────────────────────────
def remove_outliers_group_iqr(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    k: float = 1.5,
) -> pd.DataFrame:
    """
    IQR outlier removal per group.

    Phương châm: áp dụng PER SYMBOL để phát hiện biến động
    bất thường cục bộ của từng mã, tránh nhầm lẫn với biến động
    thị trường chung. Dữ liệu tài chính không chuẩn (fat-tail,
    skewed) → IQR robust hơn Z-score (Tukey, 1977).

    Parameters
    ----------
    k : float
        1.5 = tiêu chuẩn Tukey (loại nghiêm ngặt)
        2.0 / 3.0 = nới lỏng (chỉ loại cực trị)
    """
    Q1 = df.groupby(group_col)[value_col].transform(lambda x: x.quantile(0.25))
    Q3 = df.groupby(group_col)[value_col].transform(lambda x: x.quantile(0.75))
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    mask = (df[value_col] >= lower) & (df[value_col] <= upper)
    return df[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# TIME-BASED SPLIT
# ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> DataSplits:
    """
    Chronological split (KHÔNG random split).

    Phương châm: Random split với dữ liệu chuỗi thời gian
    tạo data leakage — mô hình học từ "tương lai" khi dự báo
    "quá khứ" → hiệu năng ảo. Chronological split đảm bảo
    train < val < test về mặt thời gian (Hyndman & Athanasopoulos, 2018).
    """
    n = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return DataSplits(
        train=df.iloc[:n_train],
        val=df.iloc[n_train : n_train + n_val],
        test=df.iloc[n_train + n_val :],
    )


# ─────────────────────────────────────────────────────────────────
# Z-SCORE SCALING
# ─────────────────────────────────────────────────────────────────
def scale_by_train_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Z-score standardization – FIT CHỈ TRÊN TRAIN.

    Phương châm: mean và std được tính trên train set duy nhất.
    Val và test được transform bằng cùng tham số này.
    Fit scaler trên toàn bộ data = data leakage nghiêm trọng
    (Hastie et al., 2009; Bergmeir & Benitez, 2012).

    Returns
    -------
    train_scaled, val_scaled, test_scaled, train_mean, train_std
    """
    train_mean = train_df.mean()
    train_std  = train_df.std().replace(0, 1)   # tránh chia 0

    train_scaled = (train_df - train_mean) / train_std
    val_scaled   = (val_df   - train_mean) / train_std
    test_scaled  = (test_df  - train_mean) / train_std

    return train_scaled, val_scaled, test_scaled, train_mean, train_std


# ─────────────────────────────────────────────────────────────────
# DIAGNOSTIC HELPERS
# ─────────────────────────────────────────────────────────────────
def describe_series(series: pd.Series, name: str = "") -> pd.DataFrame:
    """
    Thống kê mô tả đầy đủ cho một chuỗi thời gian.
    Bao gồm skewness và kurtosis để đánh giá phân phối.
    """
    stats = {
        "count":    series.count(),
        "mean":     series.mean(),
        "std":      series.std(),
        "min":      series.min(),
        "25%":      series.quantile(0.25),
        "50%":      series.median(),
        "75%":      series.quantile(0.75),
        "max":      series.max(),
        "skewness": series.skew(),
        "kurtosis": series.kurt(),
    }
    df = pd.DataFrame(stats, index=[name or series.name or "series"]).T
    return df.round(4)