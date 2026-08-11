from __future__ import annotations

import numpy as np
import pandas as pd


def fill_and_clean(df_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Phương châm Forward Fill:
    - Giá cổ phiếu có tính liên tục; giá ngày thiếu gần bằng giá phiên trước.
    - FFILL không tạo thông tin mới, không thay đổi xu hướng.
    - bfill() chỉ dùng dự phòng cho những ngày đầu chưa có dữ liệu.
    - Mean/Median imputation bị loại: làm mất tính động của chuỗi,
      gây sai lệch khi huấn luyện mô hình.
    """
    n_before = df_pivot.isnull().sum().sum()
    df_pivot = df_pivot.ffill().bfill()
    n_after = df_pivot.isnull().sum().sum()
    print(f"[FFILL] Missing: {n_before:,} -> {n_after}")

    invalid = (df_pivot <= 0).sum().sum()
    if invalid > 0:
        df_pivot[df_pivot <= 0] = np.nan
        df_pivot = df_pivot.ffill().bfill()
        print(f"[CLEAN] Đã xóa {invalid} giá trị <= 0")

    dup = df_pivot.index.duplicated().sum()
    if dup:
        df_pivot = df_pivot[~df_pivot.index.duplicated(keep="first")]
        print(f"[DEDUP] Loại {dup} ngày trùng")

    return df_pivot
