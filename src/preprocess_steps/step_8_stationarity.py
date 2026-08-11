from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def run_stationarity_checks(
    train_data: pd.DataFrame,
    sample_symbols: List[str],
    output_path: Path,
) -> pd.DataFrame:
    """
    Phương châm:
    - Kết hợp ADF + KPSS để có kết luận hai chiều.
    - Chỉ chạy trên tập TRAIN (không dùng val/test để tránh look-ahead bias).
    - Kết quả quyết định: ARDL dùng log-return; LSTM dùng Z-score gốc.
    """
    results = []
    for sym in sample_symbols:
        series = train_data[sym].dropna()

        try:
            adf_s, adf_p, adf_lags, _, _, _ = adfuller(series, autolag="AIC")
            adf_stat = adf_p < 0.05
        except Exception:
            adf_s, adf_p, adf_lags, adf_stat = np.nan, np.nan, np.nan, False

        try:
            kp_s, kp_p, _, _ = kpss(series, regression="c", nlags="auto")
            kp_stat = kp_p > 0.05
        except Exception:
            kp_s, kp_p, kp_stat = np.nan, np.nan, False

        if adf_stat and kp_stat:
            conclusion = "I(0) - Dừng"
        elif not adf_stat and not kp_stat:
            conclusion = "I(1)+ - Không dừng"
        elif adf_stat and not kp_stat:
            conclusion = "Trend-stationary"
        else:
            conclusion = "Diff-stationary"

        results.append(
            {
                "Symbol": sym,
                "ADF_stat": round(adf_s, 4) if not np.isnan(adf_s) else np.nan,
                "ADF_p": round(adf_p, 4) if not np.isnan(adf_p) else np.nan,
                "ADF_stationary": adf_stat,
                "KPSS_stat": round(kp_s, 4) if not np.isnan(kp_s) else np.nan,
                "KPSS_p": round(kp_p, 4) if not np.isnan(kp_p) else np.nan,
                "Conclusion": conclusion,
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

    n_stationary = (result_df["Conclusion"] == "I(0) - Dừng").sum()
    n_nonstat = result_df["Conclusion"].str.contains("Không dừng").sum()
    print(
        f"[STATIONARITY] Dừng I(0): {n_stationary}/{len(sample_symbols)} | "
        f"Không dừng: {n_nonstat}/{len(sample_symbols)}"
    )

    return result_df


def run_log_return_adf(
    train_data: pd.DataFrame,
    stationarity_df: pd.DataFrame,
    sample_symbols: List[str],
    output_path: Path,
) -> pd.DataFrame:
    """Kiểm định ADF sau khi lấy log-return để xác nhận tính dừng."""
    results = []
    for sym in sample_symbols:
        series = train_data[sym].dropna()
        log_ret = np.log(series / series.shift(1)).dropna()
        try:
            _, adf_p, _, _, _, _ = adfuller(log_ret, autolag="AIC")
            stat_after = adf_p < 0.05
        except Exception:
            adf_p, stat_after = np.nan, False

        orig_p_row = stationarity_df.loc[stationarity_df["Symbol"] == sym, "ADF_p"]
        orig_p = orig_p_row.iloc[0] if not orig_p_row.empty else np.nan

        results.append(
            {
                "Symbol": sym,
                "ADF_p_original": orig_p,
                "ADF_p_logreturn": round(adf_p, 6) if not np.isnan(adf_p) else np.nan,
                "Stationary_after_diff": stat_after,
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

    all_stat = result_df["Stationary_after_diff"].all()
    print(f"[LOG-RETURN ADF] Tất cả dừng sau diff: {all_stat}")
    return result_df
