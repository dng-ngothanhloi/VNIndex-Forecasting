from __future__ import annotations

import pickle
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# --- from step_06_export_model.py ---
def run_export_model(context: dict) -> dict:
    """Export the selected LSTM model to a pickle bundle."""
    PROJECT_ROOT = context["PROJECT_ROOT"]
    epochs = context["epochs"]
    pc_cols = context["pc_cols"]
    target_col = context["target_col"]
    x_scaler = context["x_scaler"]
    y_scaler = context["y_scaler"]
    selected_model = context["selected_model"]
    selected_metrics_row = context["selected_metrics_row"]
    summary_results = context["summary_results"]
    X_train_final = context["X_train_final"]

    # Use config-driven export selection
    _config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(_config_path, "r", encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f)
    _lstm_cfg = _cfg.get("lstm", {})

    _export_lb = _lstm_cfg.get("export_lookback", None)
    _export_bs = _lstm_cfg.get("export_batch_size", None)

    if _export_lb is not None and _export_bs is not None:
        selected_lookback   = int(_export_lb)
        selected_batch_size = int(_export_bs)
    elif selected_metrics_row is not None:
        selected_lookback   = int(selected_metrics_row['Lookback'])
        selected_batch_size = int(selected_metrics_row['Batch_size'])
    else:
        # coding-governance: no silent fallback to a stale hardcoded
        # lookback/batch_size -- if neither config export override nor a
        # populated selected_metrics_row is available, this is a pipeline-
        # ordering bug and must fail loudly.
        raise ValueError(
            "run_export_model: neither lstm.export_lookback/export_batch_size "
            "config override nor selected_metrics_row is available. Cannot "
            "determine which model to export without falling back to a "
            "stale hardcoded default."
        )

    export_dir = PROJECT_ROOT / "outputs" / "lstm_vnindex_sweep"
    export_dir.mkdir(parents=True, exist_ok=True)
    model_pkl_path = export_dir / f"lstm_vnindex_lb{selected_lookback}_bs{selected_batch_size}.pkl"

    # ── Bias correction (val residual mean) ─────────────────────────────────
    _pred_file = export_dir / f"predictions_lookback_{selected_lookback}_batch_{selected_batch_size}.csv"
    val_bias = 0.0
    if _pred_file.exists():
        _pred_df = pd.read_csv(_pred_file)
        if "Residual" in _pred_df.columns:
            val_bias = float(_pred_df["Residual"].mean())
            print(f"[BIAS] Test residual mean: {val_bias:+.4f} pts  (stored as bias_correction in bundle)")
    else:
        print("[BIAS] Prediction file not found — bias_correction=0.0")
    selected_forecast_horizon = 1
    selected_random_seed = 42
    selected_optimizer_name = "Adam"
    selected_loss_function = "mse"
    selected_scaler_type = "StandardScaler"
    # D4/P0-3B: the final deployed model is refit for exactly `best_epoch`
    # epochs on Train+Val (no EarlyStopping), not the full `epochs` sweep
    # budget. Read the actual refit epoch count when available.
    final_refit_epochs = context.get("final_refit_epochs")
    selected_train_ratio = float(_cfg.get("preprocess", {}).get("train_ratio", 0.65))

    # P0-3B/P1: Test_RMSE/Test_MAE/Test_MAPE(%) only exist on
    # `selected_metrics_row` (the final refit's genuine Test performance) --
    # `summary_results` is the TUNING-phase sweep table and has no Test
    # columns by design (Test is never touched during tuning). Prefer
    # selected_metrics_row; fall back to summary_results only for legacy
    # compatibility if it's somehow missing.
    if selected_metrics_row is not None and "Test_RMSE" in selected_metrics_row:
        selected_metrics = selected_metrics_row
    else:
        try:
            selected_row_df = summary_results.loc[(summary_results['Lookback'] == selected_lookback) & (summary_results['Batch_size'] == selected_batch_size)]
            if not selected_row_df.empty:
                selected_metrics = selected_row_df.iloc[0].to_dict()
            else:
                selected_metrics = summary_results.iloc[0].to_dict() if not summary_results.empty else {}
        except Exception:
            selected_metrics = summary_results.iloc[0].to_dict() if not summary_results.empty else {}

    model_bundle = {
        # MODEL
        "model_type": "keras_sequential_lstm",
        "model_config_json": selected_model.to_json() if selected_model is not None else None,
        "model_weights": selected_model.get_weights() if selected_model is not None else None,
        "model_config_json": selected_model.to_json() if selected_model is not None else None,
        "model_weights": selected_model.get_weights() if selected_model is not None else None,

        # HYPERPARAMETERS
        "lookback": selected_lookback,
        "batch_size": selected_batch_size,
        "epochs": final_refit_epochs if final_refit_epochs is not None else epochs,
        "sweep_epoch_budget": epochs,  # max epochs allowed during tuning (with EarlyStopping)
        "final_refit_epochs": final_refit_epochs,  # actual epochs used for the deployed model (D4, no EarlyStopping)
        "forecast_horizon": selected_forecast_horizon,

        # TRAIN CONFIG
        "optimizer_name": selected_optimizer_name,
        "loss_function": selected_loss_function,
        "random_seed": selected_random_seed,

        # DATA INFO
        "feature_columns": pc_cols,
        "target_col": target_col,
        "feature_count_per_timestep": X_train_final.shape[-1] if X_train_final is not None else None,

        # NORMALIZATION
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "scaler_type": selected_scaler_type,

        # EVALUATION
        "metrics": {
            "RMSE": selected_metrics.get("Test_RMSE"),
            "MAE": selected_metrics.get("Test_MAE"),
            "MAPE": selected_metrics.get("Test_MAPE(%)"),
        },

        # DATASET INFO
        "train_ratio": selected_train_ratio,

        # METADATA
        "selected_metrics": selected_metrics_row if selected_metrics_row else selected_metrics,
        "bias_correction": val_bias,   # add to test predictions to correct systematic under-prediction
        "exported_at": datetime.now().isoformat(),
        "version": "1.0",
    }

    with open(model_pkl_path, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"Saved LSTM model bundle to: {model_pkl_path}")
    print("Bundle contents:")
    print("- model_config_json")
    print("- model_weights")
    print("- x_scaler")
    print("- y_scaler")
    print("- lookback / batch_size / epochs / forecast_horizon")
    print("- feature metadata and selected metrics")

    context.update({
        "selected_lookback": selected_lookback,
        "selected_batch_size": selected_batch_size,
        "model_pkl_path": model_pkl_path,
        "model_bundle": model_bundle,
    })
    return context
