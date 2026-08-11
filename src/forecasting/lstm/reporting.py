from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import tensorflow.keras.backend as K
import yaml


# --- from step_07_model_summary.py ---
def run_model_summary(context: dict) -> dict:
    """Generate LSTM model summary report.

    Report field semantics (post report-consistency correction):
    A. Raw Split          -- the original Train/Val/Test chronological
       partition produced by the preprocessing split (train_df/val_df/
       test_df), BEFORE any lookback burn-in is applied.
    B. Tuning Data         -- the data actually used to fit/evaluate each
       tuning candidate for the SELECTED lookback (Train windowed with
       that lookback for fitting; the fixed 123 Val target dates for
       selection). `selected_metrics_row['Train_samples']` is the tuning
       phase's train-window sample count for this lookback, NOT "the"
       Train samples in any absolute sense -- it varies by lookback.
    C. Final Refit (Dev) Data -- the unique Train+Val development set the
       DEPLOYED model was actually refit on for `final_refit_epochs`
       epochs (selected_metrics_row['Dev_samples']/['Dev_period_*']).
    D. Final Test Data     -- the Test target dates forecast by the
       deployed model (selected_test_dates / Test_samples).
    These four populations overlap by construction (C = A.Train ∪ A.Val)
    and must never be summed into a single "Total Observations" figure.
    """
    PROJECT_ROOT = context["PROJECT_ROOT"]
    train_df = context["train_df"]
    val_df = context["val_df"]
    test_df = context["test_df"]
    pc_cols = context["pc_cols"]
    target_col = context["target_col"]
    epochs = context["epochs"]
    selected_model = context["selected_model"]
    selected_train_dates = context["selected_train_dates"]
    selected_val_dates = context["selected_val_dates"]
    selected_test_dates = context["selected_test_dates"]
    selected_history = context["selected_history"]
    selected_metrics_row = context["selected_metrics_row"]
    summary_results = context["summary_results"]

    _config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(_config_path, "r", encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f)
    _lstm_cfg = _cfg.get("lstm", {})

    # Use config-driven export selection (same logic as step_05)
    _export_lb = _lstm_cfg.get("export_lookback", None)
    _export_bs = _lstm_cfg.get("export_batch_size", None)

    if _export_lb is not None and _export_bs is not None:
        selected_lookback    = int(_export_lb)
        selected_batch_size  = int(_export_bs)
    elif selected_metrics_row is not None:
        selected_lookback    = int(selected_metrics_row['Lookback'])
        selected_batch_size  = int(selected_metrics_row['Batch_size'])
    else:
        # coding-governance: no silent fallback to a stale hardcoded
        # lookback/batch_size default.
        raise ValueError(
            "run_model_summary: neither lstm.export_lookback/export_batch_size "
            "config override nor selected_metrics_row is available."
        )
    selected_optimizer_name = "Adam"
    selected_loss_function = "mse"
    selected_scaler_type = "StandardScaler"
    selected_train_ratio = float(_cfg.get("preprocess", {}).get("train_ratio", 0.65))
    final_refit_epochs = context.get("final_refit_epochs")
    final_refit_used_early_stopping = context.get("final_refit_used_early_stopping", False)

    # ── A. RAW SPLIT (chronological Train/Val/Test partition, pre-lookback) ──
    def _period_str(idx):
        try:
            if idx is not None and len(idx) > 0:
                return f"{idx.min().strftime('%d/%m/%Y')} - {idx.max().strftime('%d/%m/%Y')}"
        except Exception:
            pass
        return "N/A"

    raw_train_period = _period_str(train_df.index if train_df is not None else None)
    raw_val_period = _period_str(val_df.index if val_df is not None else None)
    raw_test_period = _period_str(test_df.index if test_df is not None else None)
    raw_train_samples = len(train_df) if train_df is not None else 'N/A'
    raw_val_samples = len(val_df) if val_df is not None else 'N/A'
    raw_test_samples = len(test_df) if test_df is not None else 'N/A'
    try:
        total_unique_raw_observations = int(raw_train_samples) + int(raw_val_samples) + int(raw_test_samples)
    except Exception:
        total_unique_raw_observations = 'N/A'

    # ── B. TUNING DATA (for the SELECTED lookback -- varies by lookback) ──
    _tune_train_samples = selected_metrics_row.get('Train_samples') if selected_metrics_row else None
    _tune_val_samples = selected_metrics_row.get('Val_samples') if selected_metrics_row else None
    _tune_train_period = (
        f"{selected_metrics_row.get('Train_period_start')} - {selected_metrics_row.get('Train_period_end')}"
        if selected_metrics_row and selected_metrics_row.get('Train_period_start') else "N/A"
    )
    _tune_val_period = (
        f"{selected_metrics_row.get('Val_period_start')} - {selected_metrics_row.get('Val_period_end')}"
        if selected_metrics_row and selected_metrics_row.get('Val_period_start') else "N/A"
    )

    # ── C. FINAL REFIT (DEV = Train+Val) DATA (what the deployed model was
    # actually refit on) ──────────────────────────────────────────────────
    dev_period = _period_str(selected_train_dates)  # selected_train_dates == dev_dates (Train+Val)
    dev_samples = len(selected_train_dates) if selected_train_dates is not None else (
        selected_metrics_row.get('Dev_samples') if selected_metrics_row else 'N/A'
    )

    # ── D. FINAL TEST DATA (forecast by the deployed model) ──────────────
    test_period = _period_str(selected_test_dates)
    test_samples = len(selected_test_dates) if selected_test_dates is not None else 'N/A'

    # Feature list
    feature_list_text = 'PCA features: ' + ', '.join(pc_cols)

    # Learning rate and dropout
    try:
        lr = float(K.get_value(selected_model.optimizer.learning_rate))
        lr_text = f"{lr:.6f}"
    except Exception:
        lr_text = "N/A"

    try:
        drops = [l for l in selected_model.layers if 'Dropout' in l.__class__.__name__]
        dropout_rate = getattr(drops[0], 'rate', 'N/A') if drops else 'N/A'
    except Exception:
        dropout_rate = 'N/A'

    # ===== SỬA PHẦN LẤY ARCHITECTURE =====
    arch_rows = []
    try:
        if selected_model is not None:
            # Đảm bảo model đã được build
            # Lấy thông tin các layer
            for i, layer in enumerate(selected_model.layers):
                # Lấy tên layer
                layer_name = layer.__class__.__name__

                # Lấy output shape
                try:
                    if hasattr(layer, 'output_shape'):
                        out_shape = layer.output_shape
                        # Chuyển đổi None thành '?' để dễ đọc
                        if out_shape is not None:
                            if isinstance(out_shape, tuple):
                                # Xử lý tuple có None
                                out_shape_str = str(tuple('?' if dim is None else dim for dim in out_shape))
                            else:
                                out_shape_str = str(out_shape)
                        else:
                            out_shape_str = 'unknown'
                    elif hasattr(layer, 'output'):
                        out_shape = layer.output.shape
                        out_shape_str = str(tuple('?' if dim is None else dim for dim in out_shape))
                    else:
                        out_shape_str = 'unknown'
                except Exception:
                    out_shape_str = 'unknown'

                # Lấy số tham số
                try:
                    params = layer.count_params()
                except Exception:
                    params = 'N/A'

                arch_rows.append((layer_name, out_shape_str, params))

            # Lấy tổng tham số
            try:
                total_params = selected_model.count_params()
            except:
                total_params = 'N/A'

            try:
                trainable_params = int(sum([K.count_params(w) for w in selected_model.trainable_weights]))
            except:
                trainable_params = 'N/A'

            try:
                if isinstance(total_params, (int, float)):
                    if isinstance(trainable_params, (int, float)):
                        non_trainable_params = int(total_params - trainable_params)
                    else:
                        non_trainable_params = 'N/A'
                else:
                    non_trainable_params = 'N/A'
            except:
                non_trainable_params = 'N/A'
        else:
            arch_rows = []
            total_params = 'N/A'
            trainable_params = 'N/A'
            non_trainable_params = 'N/A'
    except Exception as e:
        print(f"Warning: Could not retrieve model architecture: {e}")
        arch_rows = []
        total_params = 'N/A'
        trainable_params = 'N/A'
        non_trainable_params = 'N/A'

    # Best epoch (tuning). By construction (sweep.py::run_train_and_evaluate),
    # `final_refit_epochs` IS the selected candidate's tuning-phase
    # Best_Epoch (best_epoch is threaded unchanged from selection into the
    # final refit call) -- they must never disagree. We therefore read
    # Best_Epoch directly from selected_metrics_row (never inferred from
    # final-refit history, which has no "best epoch" of its own since D4
    # final refit runs a fixed epoch count with no EarlyStopping).
    # selected_history is intentionally always None post-P1 (sweep-phase
    # Keras History objects are not kept in RAM) and is not used here.
    best_epoch = 'N/A'
    try:
        if selected_metrics_row is not None and selected_metrics_row.get('Best_Epoch') is not None:
            best_epoch = int(selected_metrics_row.get('Best_Epoch'))
    except Exception:
        pass

    # Metrics
    S_metrics = selected_metrics_row
    if S_metrics is None:
        try:
            sel_df = summary_results.loc[(summary_results['Lookback'] == selected_lookback) & (summary_results['Batch_size'] == selected_batch_size)]
            if not sel_df.empty:
                S_metrics = sel_df.iloc[0].to_dict()
            else:
                S_metrics = summary_results.iloc[0].to_dict() if not summary_results.empty else None
        except Exception:
            S_metrics = summary_results.iloc[0].to_dict() if not summary_results.empty else None

    # Print formatted report
    line = "=" * 56
    dash = "-" * 56
    print(line)
    print('                 LSTM MODEL SUMMARY')
    print(line)
    print('\nModel Type          : Sequential LSTM')
    print(f'Target Variable     : {target_col}')
    print('Forecast Horizon    : T+1')
    print('\n' + dash)
    print('DATASET INFORMATION')
    print(dash)
    print('A. RAW SPLIT (chronological Train/Val/Test partition, pre-lookback)')
    print(f'   Training Period     : {raw_train_period}')
    print(f'   Validation Period   : {raw_val_period}')
    print(f'   Testing Period      : {raw_test_period}')
    print(f'   Train Samples       : {raw_train_samples}')
    print(f'   Validation Samples  : {raw_val_samples}')
    print(f'   Test Samples        : {raw_test_samples}')
    print(f'   Total Unique Raw Observations : {total_unique_raw_observations}  '
          '(A.Train + A.Val + A.Test, disjoint, NOT the sum of B/C/D below)')
    print()
    print(f'B. TUNING DATA (for selected lookback={selected_lookback}; Train-window count '
          'varies by lookback, D3)')
    print(f'   Tuning Train Period : {_tune_train_period}')
    print(f'   Tuning Train Samples: {_tune_train_samples if _tune_train_samples is not None else "N/A"}')
    print(f'   Tuning Val Period   : {_tune_val_period}  (fixed 123 target dates for every lookback, P0-4)')
    print(f'   Tuning Val Samples  : {_tune_val_samples if _tune_val_samples is not None else "N/A"}')
    print()
    print('C. FINAL REFIT (DEV = Train+Val) DATA -- what the DEPLOYED model was actually refit on')
    print(f'   Dev Period          : {dev_period}')
    print(f'   Dev Samples         : {dev_samples}')
    print()
    print('D. FINAL TEST DATA -- forecast by the deployed model (never used for selection)')
    print(f'   Test Period         : {test_period}')
    print(f'   Test Samples        : {test_samples}')
    print('\nFeature Columns     :')
    print(feature_list_text)
    print('\nTarget Column       :', target_col)
    print('\n' + dash)
    print('HYPERPARAMETERS')
    print(dash)
    print('Look-back           :', selected_lookback)
    print('Batch Size          :', selected_batch_size)
    print('Sweep Epoch Budget  :', epochs, '(max epochs during tuning, with EarlyStopping)')
    print('Final Refit Epochs  :', final_refit_epochs if final_refit_epochs is not None else 'N/A',
          '(actual epochs used for the deployed model, D4: no EarlyStopping)')
    print('Optimizer           :', selected_optimizer_name)
    print('Loss Function       : Mean Squared Error (MSE)')
    print('\nScaler Type         :', selected_scaler_type)
    print('Dropout Rate        :', dropout_rate)
    print('Learning Rate       :', lr_text)
    print('\n' + dash)
    print('MODEL ARCHITECTURE')
    print(dash)
    print(f"{'Layer (Type)':35} {'Param #':15}")
    for name, out_shape, params in arch_rows:
        # Định dạng để hiển thị đẹp, bỏ out_shape
        print(f"{name:35} {str(params):15}")
    print('\n' + dash)
    print('Total Parameters          :', total_params)
    print('Trainable Parameters      :', trainable_params)
    print('Non-trainable Parameters  :', non_trainable_params)
    print('\n' + dash)
    print('MODEL PERFORMANCE')
    print(dash)

    def _fmt(v, pct=False):
        if v is None:
            return 'N/A'
        return f"{v:.5f}{' %' if pct else ''}"

    if S_metrics is not None:
        print('TUNING PERFORMANCE (fit on Train-only, per selected lookback -- audit only,')
        print('                     NOT the deployed model\'s fit quality; see FINAL REFIT below)')
        print('  RMSE on Train Set (tuning) :', _fmt(S_metrics.get('Train_RMSE')))
        print('  MAE on Train Set (tuning)  :', _fmt(S_metrics.get('Train_MAE')))
        print('  MAPE on Train Set (tuning) :', _fmt(S_metrics.get('Train_MAPE(%)'), pct=True))
        print('  RMSE on Val Set (selection):', _fmt(S_metrics.get('Val_RMSE')))
        print('  MAE on Val Set (selection) :', _fmt(S_metrics.get('Val_MAE')))
        print('  MAPE on Val Set (selection):', _fmt(S_metrics.get('Val_MAPE(%)'), pct=True))
        print()
        print('FINAL REFIT PERFORMANCE (deployed model, fit on Train+Val/Dev, D4)')
        print('  RMSE on Dev Set             :', _fmt(S_metrics.get('Dev_RMSE')))
        print('  MAE on Dev Set              :', _fmt(S_metrics.get('Dev_MAE')))
        print('  MAPE on Dev Set             :', _fmt(S_metrics.get('Dev_MAPE(%)'), pct=True))
        print()
        print('TEST PERFORMANCE (genuine out-of-sample, deployed model, never used for selection)')
        print('  RMSE on Test Set            :', _fmt(S_metrics.get('Test_RMSE')))
        print('  MAE on Test Set             :', _fmt(S_metrics.get('Test_MAE')))
        print('  MAPE on Test Set            :', _fmt(S_metrics.get('Test_MAPE(%)'), pct=True))
    else:
        print('No metrics available.')
    print('\n' + dash)
    print('TRAINING STATUS')
    print(dash)
    print('Training Completed       :', 'Yes' if final_refit_epochs is not None else 'N/A')
    print('Best Epoch (tuning)      :', best_epoch)
    print('Final Refit Epochs       :', final_refit_epochs if final_refit_epochs is not None else 'N/A')
    print('Early Stopping (tuning)  :', 'Enabled')
    print('Early Stopping (final refit):', 'Disabled (D4: fixed epochs=best_epoch on Train+Val)' if not final_refit_used_early_stopping else 'Enabled')
    print('\n' + line)

    return context


# --- from step_08_export_figures.py ---
def run_export_figures(context: dict) -> dict:
    """Export LSTM forecast figures to logs/figures/lstm/"""
    PROJECT_ROOT = context["PROJECT_ROOT"]
    results_dir = context["results_dir"]
    selected_pred_filename = context["selected_pred_filename"]
    selected_metrics_row = context["selected_metrics_row"]

    # ===== DINH NGHIA DUONG DAN =====
    figures_dir = PROJECT_ROOT / "logs" / "figures" / "lstm"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Figures directory: {figures_dir}")

    # ===== DOC FILE DU BAO CUA MO HINH TOT NHAT =====
    pred_file = results_dir / selected_pred_filename

    if not pred_file.exists():
        print(f"ERROR: File not found: {pred_file}")
        print("Please run step_05 first to generate predictions.")
        exit(1)

    # Derive tag for filenames from the selected model. selected_metrics_row
    # is always populated by run_train_and_evaluate() before this step runs;
    # if it is ever missing that is a pipeline-ordering bug and must fail
    # loudly rather than silently mislabel figures with a stale LB45/BS16
    # default (coding-governance: no silent fallback logic).
    if not selected_metrics_row:
        raise ValueError(
            "run_export_figures: selected_metrics_row is missing/empty. "
            "This step must run after run_train_and_evaluate() populated it; "
            "refusing to fall back to a stale hardcoded lookback/batch_size."
        )
    _sel_lb = int(selected_metrics_row['Lookback'])
    _sel_bs = int(selected_metrics_row['Batch_size'])
    model_tag = f"LB{_sel_lb}_BS{_sel_bs}"

    df = pd.read_csv(pred_file, parse_dates=['Date'])

    # Tinh sai so
    df['Residual'] = df['Actual_VNINDEX'] - df['Predicted_VNINDEX']
    df['APE_%'] = (abs(df['Residual']) / df['Actual_VNINDEX']) * 100

    # Thong ke co ban
    rmse = np.sqrt(np.mean(df['Residual']**2))
    mae = np.mean(abs(df['Residual']))
    mape = np.mean(df['APE_%'])
    std_res = np.std(df['Residual'])
    mean_res = np.mean(df['Residual'])
    skew_res = stats.skew(df['Residual'])
    kurt_res = stats.kurtosis(df['Residual'])

    print(f"\nLoaded predictions: {len(df)} samples")
    print(f"   Date range: {df['Date'].min().strftime('%d/%m/%Y')} -> {df['Date'].max().strftime('%d/%m/%Y')}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   MAPE: {mape:.4f}%")
    print(f"   Residual Mean: {mean_res:.4f}")
    print(f"   Residual Std: {std_res:.4f}")
    print(f"   Skewness: {skew_res:.4f}")
    print(f"   Kurtosis: {kurt_res:.4f}")

    print("\n" + "="*80)
    print("DANG XUAT ANH LSTM...")
    print("="*80)

    # ===== 1. BIEU DO SO SANH DU BAO =====
    print("\n  1. LSTM Forecast Plot...")
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df['Date'], df['Actual_VNINDEX'],
             label='Actual VNINDEX',
             linewidth=2,
             color='blue')
    ax1.plot(df['Date'], df['Predicted_VNINDEX'],
             label=f'LSTM Forecast (LB={_sel_lb}, BS={_sel_bs})',
             linewidth=2,
             color='red',
             linestyle='--')
    ax1.set_title(f'VNINDEX Forecast on Test Set (LSTM + PCA, LB={_sel_lb}, BS={_sel_bs})', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('VNINDEX (points)')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig1.savefig(figures_dir / f'lstm_forecast_{model_tag}.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"     Saved: lstm_forecast_{model_tag}.png")

    # ===== 2. BIEU DO PHAN DU THEO THOI GIAN =====
    print("  2. LSTM Residuals Plot...")
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(df['Date'], df['Residual'], color='blue', linewidth=1.5, label='Residuals')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=std_res, color='red', linestyle='--', linewidth=0.8, label=f'+/-1σ ({std_res:.2f})')
    ax2.axhline(y=-std_res, color='red', linestyle='--', linewidth=0.8)
    ax2.fill_between(df['Date'], -std_res, std_res, alpha=0.1, color='red')
    ax2.set_title(f'LSTM Residuals on Test Set (LB={_sel_lb}, BS={_sel_bs})', fontsize=14)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Residual (points)')
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig2.savefig(figures_dir / f'lstm_residuals_{model_tag}.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"     Saved: lstm_residuals_{model_tag}.png")

    # ===== 3. HISTOGRAM PHAN DU =====
    print("  3. LSTM Histogram...")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax3.hist(df['Residual'], bins=30, color='blue', alpha=0.7, edgecolor='black', density=True)
    # Them duong phan phoi chuan ly thuyet
    mu, sigma = np.mean(df['Residual']), np.std(df['Residual'])
    x = np.linspace(df['Residual'].min(), df['Residual'].max(), 100)
    ax3.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
    ax3.axvline(x=0, color='green', linestyle='-', linewidth=1.5, label=f'Mean = {mu:.2f}')
    ax3.set_title(f'Histogram of LSTM Residuals (LB={_sel_lb}, BS={_sel_bs})', fontsize=14)
    ax3.set_xlabel('Residual (points)')
    ax3.set_ylabel('Density')
    ax3.legend(loc='best')
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    fig3.savefig(figures_dir / f'lstm_histogram_{model_tag}.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"     Saved: lstm_histogram_{model_tag}.png")

    # ===== 4. QQ-PLOT =====
    print("  4. LSTM QQ-Plot...")
    fig4, ax4 = plt.subplots(figsize=(8, 8))
    stats.probplot(df['Residual'], dist="norm", plot=ax4)
    ax4.set_title(f'Q-Q Plot of LSTM Residuals (LB={_sel_lb}, BS={_sel_bs})', fontsize=14)
    ax4.grid(alpha=0.3)
    plt.tight_layout()
    fig4.savefig(figures_dir / f'lstm_qqplot_{model_tag}.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print(f"     Saved: lstm_qqplot_{model_tag}.png")

    # ===== 5. SCATTER PLOT =====
    print("  5. LSTM Scatter Plot...")
    fig5, ax5 = plt.subplots(figsize=(8, 8))
    ax5.scatter(df['Actual_VNINDEX'], df['Predicted_VNINDEX'], alpha=0.5, color='blue', s=30)
    min_val = min(df['Actual_VNINDEX'].min(), df['Predicted_VNINDEX'].min())
    max_val = max(df['Actual_VNINDEX'].max(), df['Predicted_VNINDEX'].max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    # Them R2
    corr = np.corrcoef(df['Actual_VNINDEX'], df['Predicted_VNINDEX'])[0, 1]
    r2 = corr**2
    ax5.text(0.05, 0.95, f'R2 = {r2:.4f}', transform=ax5.transAxes, fontsize=12, verticalalignment='top')
    ax5.set_xlabel('Actual VNINDEX (points)')
    ax5.set_ylabel('Predicted VNINDEX (points)')
    ax5.set_title(f'Actual vs Predicted VNINDEX (LSTM, LB={_sel_lb}, BS={_sel_bs})', fontsize=14)
    ax5.legend(loc='best')
    ax5.grid(alpha=0.3)
    plt.tight_layout()
    fig5.savefig(figures_dir / f'lstm_scatter_{model_tag}.png', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    print(f"     Saved: lstm_scatter_{model_tag}.png")

    # ===== 6. BOXPLOT PHAN DU =====
    print("  6. LSTM Boxplot...")
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    bp = ax6.boxplot(df['Residual'], vert=True, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', color='blue'),
                      whiskerprops=dict(color='blue'),
                      capprops=dict(color='blue'),
                      medianprops=dict(color='red', linewidth=2))
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.set_title(f'Boxplot of LSTM Residuals (LB={_sel_lb}, BS={_sel_bs})', fontsize=14)
    ax6.set_ylabel('Residual (points)')
    ax6.set_xticklabels(['Residuals'])
    ax6.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    fig6.savefig(figures_dir / f'lstm_boxplot_{model_tag}.png', dpi=300, bbox_inches='tight')
    plt.close(fig6)
    print(f"     Saved: lstm_boxplot_{model_tag}.png")

    # ===== TONG KET =====
    print("\n" + "="*80)
    print(f"All 6 figures saved to: {figures_dir}")
    print("="*80)
    print("\nFigure files:")
    print(f"   lstm_forecast_{model_tag}.png      (Forecast comparison)")
    print(f"   lstm_residuals_{model_tag}.png     (Residuals over time)")
    print(f"   lstm_histogram_{model_tag}.png     (Histogram of residuals)")
    print(f"   lstm_qqplot_{model_tag}.png        (Q-Q plot)")
    print(f"   lstm_scatter_{model_tag}.png       (Actual vs Predicted)")
    print(f"   lstm_boxplot_{model_tag}.png       (Boxplot of residuals)")
    print("="*80)

    # ===== LUU FILE CSV THONG KE PHAN DU =====
    stats_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'MAPE', 'Mean_Residual', 'Std_Residual', 'Min_Residual', 'Max_Residual', 'Skewness', 'Kurtosis'],
        'Value': [rmse, mae, mape, mean_res, std_res, df['Residual'].min(), df['Residual'].max(), skew_res, kurt_res]
    })
    stats_file = results_dir / f"residual_statistics_lb{_sel_lb}_bs{_sel_bs}.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"\nSaved residual statistics to: {stats_file}")

    context["figures_dir"] = figures_dir
    return context
