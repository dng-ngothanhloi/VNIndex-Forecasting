from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import remove_outliers_group_iqr


def remove_outliers_with_log(
    stocks_df: pd.DataFrame,
    symbol_col: str,
    close_col: str,
    k: float,
    output_path: Path,
) -> pd.DataFrame:
    """
    Phương châm:
    - Dữ liệu tài chính không tuân phân phối chuẩn (skewed, fat-tail)
      -> IQR là lựa chọn robust, không cần giả định phân phối.
    - Áp dụng PER SYMBOL (không phải toàn thị trường) để phát hiện
      biến động bất thường CỤC BỘ của từng mã, không loại nhầm các
      giai đoạn thị trường chung tăng/giảm mạnh.
    - Log ngoại lai lưu artifact để truy vết.
    """
    stocks_clean = remove_outliers_group_iqr(
        stocks_df, group_col=symbol_col, value_col=close_col, k=k
    )

    removed_mask = ~stocks_df.index.isin(stocks_clean.index)
    if removed_mask.sum() > 0:
        outlier_log = (
            stocks_df[removed_mask]
            .groupby(symbol_col)
            .agg(
                n_outliers=(close_col, "count"),
                min_val=(close_col, "min"),
                max_val=(close_col, "max"),
                mean_val=(close_col, "mean"),
            )
            .reset_index()
            .sort_values("n_outliers", ascending=False)
        )
    else:
        outlier_log = pd.DataFrame(
            columns=[symbol_col, "n_outliers", "min_val", "max_val", "mean_val"]
        )

    outlier_log.to_csv(output_path, index=False)

    n_removed = len(stocks_df) - len(stocks_clean)
    print(
        f"[IQR] k={k} | Trước: {len(stocks_df):,} | Sau: {len(stocks_clean):,} "
        f"| Loại: {n_removed:,} ({n_removed / len(stocks_df) * 100:.2f}%)"
    )
    print(f"[IQR] Log saved -> {output_path}")

    return stocks_clean
