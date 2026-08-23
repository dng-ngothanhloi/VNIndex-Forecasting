"""
reporting.py – ARDL model summary reporting and visualization
==============================================================
Steps 08-11 of the ARDL pipeline:

  08 run_summary        Score Board: model spec, observation-count
                        disambiguation, Val OOS / Train+Val / Test metrics,
                        residual diagnostics, statsmodels coefficient table
  09 run_plot           5 evaluation figures (forecast, residuals, QQ,
                        histogram, actual-vs-predicted scatter)
  10 run_ardl_80obs     Short-window (recent-period) diagnostic subset
  11 run_summary_table  Full statsmodels-style regression results table
                        including coefficient CIs and residual diagnostics

Restored from reporting_v0.1.py (the canonical pre-loss implementation),
which already carries the report-consistency corrections: explicit
"Fixed-coefficient rolling one-step-ahead" terminology, causal/hold_back
disclosure, and the Development-input vs effective-nobs vs Test-predictions
distinction that must never be conflated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from tabulate import tabulate

from .common import mape, paired_valid, rmse
from sklearn.metrics import mean_absolute_error, r2_score


# --- from step_08_summary.py ---
def run_summary(context: dict) -> dict:
    ardl_model = context["ardl_model"]
    ardl_res = context["ardl_res"]
    metrics = context["metrics"]
    diag = context["diag"]
    forecast_path = context["forecast_path"]

    # ── Data periods ──────────────────────────────────────────────
    y_train    = context.get("y_train",    context["y_trainval"])
    y_val      = context.get("y_val",      None)
    y_trainval = context["y_trainval"]
    y_test     = context["y_test"]

    train_period = f"{y_train.index.min().date()} → {y_train.index.max().date()}"
    val_period   = (f"{y_val.index.min().date()} → {y_val.index.max().date()}"
                    if y_val is not None else "N/A")
    test_period  = f"{y_test.index.min().date()} → {y_test.index.max().date()}"

    causal = bool(context.get("ardl_causal", True))
    hold_back = context.get("ardl_hold_back")

    print(f"Selected pair: {context['SELECTED_PAIR']}")
    print("Model type: Fixed-coefficient rolling one-step-ahead ARDL with PCA exogenous variables")
    print("Forecast Horizon: T+1 (rolling one-step-ahead; NOT multi-step recursive forecast)")
    print(f"  causal={causal}  (True => no contemporaneous PC.L0; only t-1 and earlier lags predict y(t))")
    print(f"  hold_back={hold_back}  (fixed across candidates for fair IC comparison)")
    print(f"  AR lags (p):", max(ardl_model._lags) if ardl_model._lags else 0)
    print(f"  AR lag list :", ardl_model._lags)
    print(f"  Exog lags map:", ardl_model._order)
    print(f"  Number of parameters:", len(ardl_res.params))
    print()
    print("  Data periods:")
    print(f"    Train      : {train_period}  (n={len(y_train)})")
    print(f"    Validation : {val_period}" + (f"  (n={len(y_val)})" if y_val is not None else ""))
    print(f"    Test       : {test_period}  (n={len(y_test)})")
    print()
    print("  NOTE on observation counts (do not conflate these):")
    print(f"    Development input observations (Train+Val rows fed to ARDL): {len(y_trainval)}")
    print(f"    Effective estimation nobs (statsmodels, after hold_back={hold_back} burn-in): "
          f"{int(ardl_res.nobs)}")
    print(f"    Test predictions (rolling one-step-ahead, out-of-sample): {len(y_test)}")
    print()

    print("  statsmodels native model summary (its own 'No. Observations' reflects the")
    print("  EFFECTIVE estimation sample after hold_back burn-in, shown here unmodified):")
    print(ardl_res.summary().tables[1])

    # Val OOS metric (from sweep table if available)
    sweep_table = context.get("ardl_sweep_table")
    if sweep_table is not None:
        p, q = context["SELECTED_PAIR"]
        row = sweep_table[(sweep_table["P"] == p) & (sweep_table["Q"] == q)]
        if not row.empty and "RMSE_val" in row.columns:
            rmse_val = row["RMSE_val"].iloc[0]
            mae_val  = row.get("MAE_val", row.get("MAE_val", None))
            print(f"RMSE Val (OOS, train-only fit) : {rmse_val:.6f}")
            if mae_val is not None and not row["MAE_val"].isna().all():
                print(f"MAE  Val (OOS, train-only fit) : {row['MAE_val'].iloc[0]:.6f}")

    print(f"RMSE tren tap Train+Val : {metrics['RMSE_trainval']:.6f}")
    print(f"RMSE tren tap Test      : {metrics['RMSE_test']:.6f}")
    print(f"MAE tren tap Test       : {metrics['MAE_test']:.6f}")
    print(f"MAPE tren tap Test (%)  : {metrics['MAPE_test(%)']:.6f}")
    print(f"R2 tren tap Test        : {metrics['R2_test']:.6f}")
    print("=" * 70)
    print(f"Forecast file           : {forecast_path}")

    print("\nDiagnostics:")
    for name, value in diag.items():
        print(f"  {name}: {value:.6f}")

    context["summary_displayed"] = True
    return context


# --- from step_09_plot.py ---
def run_plot(context: dict) -> dict:
    y_test = context["y_test"]
    pred_test = context["pred_test"]
    selected_pair = context["SELECTED_PAIR"]

    # Lấy figures_dir từ context (đã tạo ở step 1)
    figures_dir = context.get("figures_dir")

    # Nếu chưa có thì tạo mới (phòng trường hợp step 1 chưa chạy)
    if figures_dir is None:
        figures_dir = Path("logs/figures/ardl")
        figures_dir.mkdir(parents=True, exist_ok=True)
        context["figures_dir"] = figures_dir

    # ===== ĐẢM BẢO THƯ MỤC TỒN TẠI TRƯỚC KHI LƯU =====
    figures_dir.mkdir(parents=True, exist_ok=True)

    resid = y_test.values - pred_test.values

    print("\n" + "="*70)
    print(" ARDL STEP 9: ĐANG XUẤT ẢNH...")
    print("="*70)
    print(f" Lưu vào: {figures_dir.resolve()}")

    # ===== ẢNH 1: Actual vs Predicted =====
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(y_test.index, y_test.values, label="Actual VNINDEX", linewidth=2, color='blue')
    ax1.plot(pred_test.index, pred_test.values, label=f"ARDL Forecast (p={selected_pair[0]}, q={selected_pair[1]})",
             linewidth=2, color='red', linestyle='--')
    ax1.set_title(f"VNINDEX Forecast on Test Set (ARDL + PCA)", fontsize=14)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("VNINDEX")
    ax1.legend()
    ax1.grid(alpha=0.3)

    fig1_path = figures_dir / f"ardl_forecast_P{selected_pair[0]}_Q{selected_pair[1]}.png"
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"   Saved: {fig1_path.name}")

    # ===== ẢNH 2: Residuals theo thời gian =====
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(y_test.index, resid, color='red', linewidth=1, label='Residual')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.axhline(y=np.std(resid), color='gray', linestyle=':', linewidth=0.8, label=f'±1σ ({np.std(resid):.2f})')
    ax2.axhline(y=-np.std(resid), color='gray', linestyle=':', linewidth=0.8)
    ax2.set_title(f"ARDL Residuals on Test Set (pair={selected_pair})", fontsize=14)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Residual")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig2_path = figures_dir / f"ardl_residuals_P{selected_pair[0]}_Q{selected_pair[1]}.png"
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"   Saved: {fig2_path.name}")

    # ===== ẢNH 3: QQ-plot =====
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    stats.probplot(resid, dist="norm", plot=ax3)
    ax3.set_title("QQ-plot of ARDL Residuals", fontsize=14)

    fig3_path = figures_dir / f"ardl_qqplot_P{selected_pair[0]}_Q{selected_pair[1]}.png"
    fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"   Saved: {fig3_path.name}")

    # ===== ẢNH 4: Histogram residuals =====
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.hist(resid, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Mean = 0')
    ax4.axvline(x=np.mean(resid), color='green', linestyle='--', linewidth=1.5, label=f'Mean = {np.mean(resid):.2f}')
    ax4.set_title("Histogram of ARDL Residuals", fontsize=14)
    ax4.set_xlabel("Residual")
    ax4.set_ylabel("Frequency")
    ax4.legend()
    ax4.grid(alpha=0.3)

    fig4_path = figures_dir / f"ardl_histogram_P{selected_pair[0]}_Q{selected_pair[1]}.png"
    fig4.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print(f"   Saved: {fig4_path.name}")

    # ===== ẢNH 5: Actual vs Predicted scatter =====
    fig5, ax5 = plt.subplots(figsize=(6, 6))
    ax5.scatter(y_test.values, pred_test.values, alpha=0.6, color='steelblue')
    ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect fit')
    ax5.set_xlabel("Actual VNINDEX")
    ax5.set_ylabel("Predicted VNINDEX")
    ax5.set_title(f"Actual vs Predicted (Test Set)\nR² = {context['metrics']['R2_test']:.4f}", fontsize=12)
    ax5.legend()
    ax5.grid(alpha=0.3)

    fig5_path = figures_dir / f"ardl_scatter_P{selected_pair[0]}_Q{selected_pair[1]}.png"
    fig5.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close(fig5)
    print(f"   Saved: {fig5_path.name}")

    print(f"\n All {5} figures saved to: {figures_dir.resolve()}")
    print("="*70 + "\n")

    context["figures_dir"] = figures_dir
    return context


# --- from step_10_ardl_80obs.py ---
def run_ardl_80obs(context: dict) -> dict:
    """
    Tính kết quả ARDL trên 80 quan sát (30/12/2024 - 29/04/2025)
    """
    selected_pair = context.get("SELECTED_PAIR", (5, 2))
    y_test = context["y_test"]
    pred_test = context["pred_test"]

    # Lọc dữ liệu từ 30/12/2024
    start_date_80 = pd.Timestamp('2024-12-30')
    mask_80 = y_test.index >= start_date_80

    y_test_80 = y_test[mask_80]
    pred_test_80 = pred_test[mask_80]

    # Guard: this is a dataset-specific recent-window diagnostic. If the
    # Test split does not extend past start_date_80 (e.g. a different
    # dataset or split), skip rather than crash -- it is diagnostic only
    # and never feeds model selection or the canonical Test metrics.
    if len(y_test_80) == 0:
        print(f"[ARDL Step 10] No Test observations on/after "
              f"{start_date_80.date()} — 80-obs diagnostic skipped.")
        context.update({"metrics_80": None, "forecast_80_path": None, "forecast_80": None})
        return context

    y_test_eval, pred_test_eval = paired_valid(y_test_80, pred_test_80)

    # Tính metrics
    metrics_80 = {
        "n_obs": len(y_test_eval),
        "RMSE": rmse(y_test_eval, pred_test_eval),
        "MAE": float(mean_absolute_error(y_test_eval, pred_test_eval)),
        "MAPE": mape(y_test_eval, pred_test_eval),
        "R2": float(r2_score(y_test_eval, pred_test_eval)),
    }

    # NOTE: paired_valid() returns plain numpy arrays (no DatetimeIndex),
    # so build the forecast table from the pre-filter Series (y_test_80 /
    # pred_test_80) which still carry the original dates.
    forecast_80 = pd.DataFrame({
        "Date": y_test_80.index,
        "Actual_VNINDEX": y_test_80.values,
        "Predicted_VNINDEX": pred_test_80.values,
        "Residual": y_test_80.values - pred_test_80.values,
    })

    results_dir = context["PROJECT_ROOT"] / "outputs" / "ardl_vnindex_forecast"
    results_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = results_dir / f"ardl_test_forecast_80obs_P{selected_pair[0]}_Q{selected_pair[1]}.csv"
    forecast_80.to_csv(forecast_path, index=False)
    print(f"\n[COMP] Da luu ket qua vao: {forecast_path}")
    print(f"  [Step 10] Recent-window diagnostic (n={metrics_80['n_obs']}): "
          f"RMSE={metrics_80['RMSE']:.4f} MAE={metrics_80['MAE']:.4f} R2={metrics_80['R2']:.4f}")
    print(f"  NOTE: this is a DIAGNOSTIC subset of Test, not the canonical Test metric "
          f"(canonical Test n={len(y_test)}). Never use it for model selection or "
          f"cross-representation comparison.")

    context.update({
        "metrics_80": metrics_80,
        "forecast_80_path": forecast_path,
        "forecast_80": forecast_80,
    })

    return context


# --- from step_11_summary_table.py ---
def run_summary_table(context: dict) -> dict:
    """In bảng tóm tắt mô hình ARDL + PCA"""

    res = context["ardl_res"]
    selected_pair = context["SELECTED_PAIR"]
    metrics = context["metrics"]
    diag = context["diag"]

    print("\nTóm tắt mô hình ARDL:")
    print("\n" + "="*90)
    print("                            ARDL + PCA Regression Results")
    print("="*90)

    # ===== 1. THÔNG TIN MÔ HÌNH =====
    _dev_input_n = len(context["y_trainval"])
    _effective_nobs = int(res.nobs)
    info_data = [
        ["Dep. Variable:", "VNINDEX", "No. Observations:", _dev_input_n],
        ["Model:", f"ARDL({selected_pair[0]},{selected_pair[1]})+PCA", "Log Likelihood:", f"{res.llf:.3f}"],
        ["Date:", pd.Timestamp.now().strftime("%a, %d %b %Y"), "AIC:", f"{res.aic:.3f}"],
        ["Time:", pd.Timestamp.now().strftime("%H:%M:%S"), "BIC:", f"{res.bic:.3f}"],
        ["Sample:", f"0 - {len(context['y_trainval'])-1}", "HQIC:", f"{res.hqic:.3f}"],
        ["Covariance Type:", "nonrobust", "", ""]
    ]

    print(tabulate(info_data, tablefmt="plain", numalign="left", stralign="left"))
    print(f"\n  [Project note -- this Score Board table only, statsmodels' own summary is untouched]")
    print(f"  No. Observations above = Development INPUT rows fed to ARDL (Train+Val) = {_dev_input_n}")
    print(f"  Effective ARDL estimation nobs (after hold_back burn-in, statsmodels-reported) = {_effective_nobs}")
    print(f"  Test predictions (rolling one-step-ahead, out-of-sample, separate population) = {len(context['y_test'])}")

    # ===== 2. HỆ SỐ HỒI QUY =====
    print("\n" + "="*90)
    print("                 coef    std err        t      P>|t|      [0.025      0.975]")
    print("-"*90)

    # Lấy tên biến (loại bỏ khoảng trắng thừa)
    for idx, param in enumerate(res.params.index):
        param_name = param.strip()
        coef_val = res.params.iloc[idx]
        se_val = res.bse.iloc[idx]
        t_val = res.tvalues.iloc[idx]
        p_val = res.pvalues.iloc[idx]
        ci_lower = coef_val - 1.96 * se_val
        ci_upper = coef_val + 1.96 * se_val

        # Format theo đúng style của statsmodels
        print(f"{param_name:>18} {coef_val:10.4f} {se_val:10.4f} {t_val:9.3f} {p_val:9.3f} {ci_lower:10.4f} {ci_upper:10.4f}")

    # ===== 3. THỐNG KÊ PHẦN DƯ =====
    print("\n" + "="*90)

    # Lấy residuals từ context
    forecast_table = context.get("forecast_table")
    if forecast_table is not None and "Residual" in forecast_table.columns:
        resid = forecast_table["Residual"]
        resid_arr = np.array(resid.dropna())

        # Tính các thống kê phần dư
        jb_stat = diag.get('JarqueBera', 0)
        jb_p = diag.get('JB_pvalue', 0)
        skew = diag.get('Skew', 0)
        kurt = diag.get('Kurtosis', 0)

        # Lấy LB test từ diag (có thể được lưu từ step 9)
        lb_q1 = diag.get('LjungBox_Q_L1', 0)
        lb_p1 = diag.get('LjungBox_p_L1', 0)

        arch_stat = diag.get('ARCH_stat', 0)
        arch_p = diag.get('ARCH_pvalue', 0)

        # In thống kê dạng mẫu
        print(f"Ljung-Box (L1) (Q):       {lb_q1:>8.2f}        Jarque-Bera (JB):      {jb_stat:>8.2f}")
        print(f"Prob(Q):                   {lb_p1:>8.3f}        Prob(JB):               {jb_p:>8.3f}")
        print(f"ARCH LM:                   {arch_stat:>8.2f}        Prob(ARCH):             {arch_p:>8.3f}")
        print(f"Skew:                      {skew:>8.3f}        Kurtosis:               {kurt:>8.3f}")

    return context
