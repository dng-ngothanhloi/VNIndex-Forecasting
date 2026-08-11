from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def analyze_correlation(
    df_pivot: pd.DataFrame,
    processed_dir: Path,
) -> pd.DataFrame:
    """
    Phương châm: Phân tích tương quan TRƯỚC PCA để xác nhận đa cộng tuyến
    nghiêm trọng -> biện hộ cho việc áp dụng PCA.
    Kết quả được lưu artifact và trích bảng vào báo cáo.
    """
    corr_matrix = df_pivot.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    corr_vals = upper.stack()

    thresholds = [0.50, 0.70, 0.80, 0.90]
    rows = []
    total_pairs = len(corr_vals)
    for thr in thresholds:
        cnt = (corr_vals.abs() > thr).sum()
        rows.append(
            {
                "Ngưỡng |r|": f"> {thr:.2f}",
                "Số cặp": cnt,
                "Tỷ lệ (%)": f"{cnt / total_pairs * 100:.1f}",
            }
        )

    corr_summary = pd.DataFrame(rows)
    corr_summary.to_csv(processed_dir / "corr_summary.csv", index=False)

    print("[CORRELATION ANALYSIS]")
    print(f"  Tổng cặp: {total_pairs:,} | Trung bình |r|: {corr_vals.abs().mean():.4f}")
    print(corr_summary.to_string(index=False))

    corr_matrix.to_csv(processed_dir / "corr_matrix.csv")

    return corr_summary
