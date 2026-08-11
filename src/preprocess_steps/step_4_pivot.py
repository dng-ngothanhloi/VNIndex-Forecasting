from __future__ import annotations

import pandas as pd


def pivot_to_wide(
    stocks_clean: pd.DataFrame,
    date_col: str,
    symbol_col: str,
    close_col: str,
) -> pd.DataFrame:
    """
    Chuyển Long format -> Wide format (ma trận ngày × mã).
    Đây là định dạng đầu vào chuẩn cho PCA và LSTM.
    """
    stocks_clean = stocks_clean.drop_duplicates(subset=[date_col, symbol_col])
    df_pivot = stocks_clean.pivot(index=date_col, columns=symbol_col, values=close_col)
    df_pivot.columns.name = None
    print(f"[PIVOT] Wide shape: {df_pivot.shape}  (ngày × mã)")
    return df_pivot
