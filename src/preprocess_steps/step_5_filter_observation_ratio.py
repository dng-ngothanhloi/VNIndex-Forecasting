from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def filter_by_observation_ratio(
    df_pivot: pd.DataFrame,
    threshold: float,
    core_dir: Path,
    quality_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Phương châm:
        - Mã có <80% quan sát thường là: mới niêm yết (<2020), đang bị kiểm
            soát giao dịch (suspended), thanh khoản quá thấp -> không liên quan
            đến biến động VN-Index.
        - Tổng ngày giao dịch lý thuyết 5 năm ~1.250 ngày (sau bỏ T7/CN, lễ).
            Mã đạt >=80% = >=1.000 ngày -> đủ lịch sử cho mô hình học.
        - Nội suy dài hạn (>20% missing) tạo thông tin giả, làm sai cấu trúc
            phương sai -> ảnh hưởng trực tiếp đến kết quả PCA.
        """
        missing_ratio = df_pivot.isnull().mean()

        bins = [
                ("0% (đầy đủ)", missing_ratio == 0),
                ("0-5%", (missing_ratio > 0) & (missing_ratio <= 0.05)),
                ("5-10%", (missing_ratio > 0.05) & (missing_ratio <= 0.10)),
                ("10-20%", (missing_ratio > 0.10) & (missing_ratio <= 0.20)),
                (f">{threshold*100:.0f}% (loại)", missing_ratio > threshold),
        ]
        dist_rows = [{"Khoảng missing": label, "Số mã": cond.sum()} for label, cond in bins]
        missing_dist = pd.DataFrame(dist_rows)
        missing_dist.to_csv(quality_dir / "missing_dist.csv", index=False)
        print("[MISSING DISTRIBUTION]")
        print(missing_dist.to_string(index=False))

        valid_cols = missing_ratio[missing_ratio <= threshold].index.tolist()
        removed_cols = missing_ratio[missing_ratio > threshold].index.tolist()

        removed_df = pd.DataFrame(
                {
                        "Symbol": removed_cols,
                        "missing_ratio": [missing_ratio[s] for s in removed_cols],
                        "reason": "missing_ratio > threshold",
                }
        ).sort_values("missing_ratio", ascending=False)
        removed_df.to_csv(core_dir / "removed_stocks.csv", index=False)

        pd.Series(valid_cols, name="Symbol").to_csv(core_dir / "valid_stocks.csv", index=False)

        print(f"[FILTER 80%] Giữ lại: {len(valid_cols)} | Loại bỏ: {len(removed_cols)}")
        return df_pivot[valid_cols].copy(), missing_dist
