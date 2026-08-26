from __future__ import annotations

import random
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .data import make_windowed_data, add_target_history, make_cross_boundary_windowed_data


# --- from step_05_train_and_evaluate.py ---
def get_adaptive_patience(use_overlap: bool, base: int, multiplier: float = 2.0) -> int:
    """Increase patience when using overlap validation to avoid premature stopping."""
    if use_overlap:
        return int(base * multiplier)
    return base


class MinEpochEarlyStopping(keras.callbacks.EarlyStopping):
    """EarlyStopping that refuses to fire before `start_epoch` completed epochs.

    Rationale: with overlap validation, val_loss can plateau very early
    (model sees data resembling val during training) which would trigger
    EarlyStopping before the model has had a fair chance to converge.
    This is a HARD guarantee, not just a post-hoc warning.
    """

    def __init__(self, start_epoch: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        # `epoch` is 0-indexed; skip monitor logic until start_epoch reached
        if epoch < self.start_epoch:
            return
        super().on_epoch_end(epoch, logs)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred):
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


def _build_lstm_model(lookback: int, n_features: int, *, learning_rate: float,
                       lstm_units, dense_units, dropout_rate: float,
                       use_batch_norm: bool, seed: int = 42):
    """Build LSTM model with deterministic weights (unchanged architecture)."""
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    layer_list = [layers.Input(shape=(lookback, n_features))]

    layer_list.append(layers.LSTM(lstm_units[0], return_sequences=True))
    if use_batch_norm:
        layer_list.append(layers.BatchNormalization())
    layer_list.append(layers.Dropout(dropout_rate))

    layer_list.append(layers.LSTM(lstm_units[1]))
    if use_batch_norm:
        layer_list.append(layers.BatchNormalization())
    layer_list.append(layers.Dropout(dropout_rate))

    layer_list.append(layers.Dense(dense_units[0], activation="relu"))
    layer_list.append(layers.Dense(1))

    model = keras.Sequential(layer_list)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    if use_batch_norm:
        print("[CONFIG] BatchNormalization enabled")
    return model


def final_refit_and_forecast(
    *,
    train_scaled_df: pd.DataFrame,
    val_scaled_df: pd.DataFrame,
    test_scaled_df: pd.DataFrame,
    pc_cols: list,
    target_col: str,
    y_scaler,
    selected_lookback: int,
    selected_batch_size: int,
    best_epoch: int,
    learning_rate: float,
    lstm_units,
    dense_units,
    dropout_rate: float,
    use_batch_norm: bool,
    seed: int = 42,
) -> dict:
    """Reusable final-refit-and-forecast step (P0-3B/D4), factored out of
    run_train_and_evaluate so Phase 3D multi-seed stability can call it
    once per seed WITHOUT re-running the tuning sweep or re-selecting
    hyperparameters. `selected_lookback`/`selected_batch_size`/`best_epoch`
    must already be frozen (e.g. from a single seed=42 reference tuning
    run) before this function is called.

    Instantiates a NEW model (seeded with `seed`), refits on the unique
    Train+Val development set for exactly `best_epoch` epochs (no
    EarlyStopping/ReduceLROnPlateau), and forecasts ALL Test targets via
    cross-boundary windowing. Returns a dict with the fitted model,
    predictions, dates, and metrics -- Test is used here ONLY for
    reporting genuine out-of-sample performance, never for any
    hyperparameter/epoch selection (that already happened upstream).
    """
    trainval_scaled_df = pd.concat([train_scaled_df, val_scaled_df], axis=0).sort_index()
    if trainval_scaled_df.index.has_duplicates:
        raise AssertionError("Train+Val development set has duplicate dates — split boundary violated.")

    X_dev, y_dev, dev_dates = make_windowed_data(trainval_scaled_df, pc_cols, target_col, selected_lookback)
    X_dev_hist = add_target_history(trainval_scaled_df, target_col, selected_lookback)
    X_dev_final = np.concatenate([X_dev, X_dev_hist], axis=2)

    final_model = _build_lstm_model(
        selected_lookback, X_dev_final.shape[-1],
        learning_rate=learning_rate, lstm_units=lstm_units,
        dense_units=dense_units, dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm, seed=seed,
    )
    final_model.fit(
        X_dev_final, y_dev,
        epochs=best_epoch, batch_size=selected_batch_size,
        verbose=0, shuffle=False,
    )

    X_test, y_test, X_test_hist, test_dates = make_cross_boundary_windowed_data(
        trainval_scaled_df, test_scaled_df, pc_cols, target_col, selected_lookback,
    )
    X_test_final = np.concatenate([X_test, X_test_hist], axis=2)
    assert len(X_test_final) == len(test_scaled_df), (
        f"P0-4 violation: Test samples ({len(X_test_final)}) != len(test_scaled_df) "
        f"({len(test_scaled_df)}) for selected lookback={selected_lookback}."
    )

    test_pred = y_scaler.inverse_transform(final_model.predict(X_test_final, verbose=0)).ravel()
    y_test_true = y_scaler.inverse_transform(y_test).ravel()

    dev_pred = y_scaler.inverse_transform(final_model.predict(X_dev_final, verbose=0)).ravel()
    y_dev_true = y_scaler.inverse_transform(y_dev).ravel()

    metrics = {
        "Test_RMSE": rmse(y_test_true, test_pred),
        "Test_MAE": float(mean_absolute_error(y_test_true, test_pred)),
        "Test_MAPE(%)": mape(y_test_true, test_pred),
        "Dev_RMSE": rmse(y_dev_true, dev_pred),
        "Dev_MAE": float(mean_absolute_error(y_dev_true, dev_pred)),
        "Dev_MAPE(%)": mape(y_dev_true, dev_pred),
    }

    return {
        "model": final_model,
        "seed": seed,
        "test_dates": test_dates,
        "y_test_true": y_test_true,
        "test_pred": test_pred,
        "dev_dates": dev_dates,
        "metrics": metrics,
        "X_dev_final": X_dev_final,
    }


def prepare_target_history_context(context: dict) -> dict:
    """Derive a target-history-only context from an existing LSTM context.

    The goal is to produce a model whose input tensor is:
        shape = (L, 1)  — VNINDEX history only, no PCA channels

    The canonical sweep loop always concatenates two arrays:
        X          from make_windowed_data(df, pc_cols, ...)   shape (N, L, len(pc_cols))
        X_hist     from add_target_history(df, ...)            shape (N, L, 1)
        X_final  = concat([X, X_hist], axis=2)                 shape (N, L, len(pc_cols)+1)

    To make the final tensor (N, L, 1) without touching the sweep loop,
    we set pc_cols=[] (empty) so X has shape (N, L, 0), and store the
    VNINDEX series as the target_col inside the scaled DataFrames so
    add_target_history returns (N, L, 1).  The concat then gives
    (N, L, 0+1) = (N, L, 1). ✓

    This is the cleanest zero-duplication approach: the VNINDEX history
    arrives exactly once in the final input tensor, matching the
    AR(1) → PCA-ARDL analogy where AR(1) uses only y(t-1).

    Scientific comparison enabled:
      TH-LSTM   input shape (L, 1): VNINDEX history only
      PCA-LSTM  input shape (L, k+1): k PC channels + VNINDEX history
      Question: does adding k PC channels improve forecast?

    Context keys changed:
      pc_cols          → []   (empty list — no feature channels)
      train/val/test   → DataFrames containing only the target_col column
      _th_mode         → True (flag for callers / artifact labelling)
    """
    import copy

    train_scaled_df = context["train_scaled_df"]
    val_scaled_df   = context["val_scaled_df"]
    test_scaled_df  = context["test_scaled_df"]
    target_col      = context["target_col"]

    # Keep only the target column. make_windowed_data(df, pc_cols=[], ...)
    # will produce X of shape (N, L, 0); add_target_history will then
    # contribute the sole (N, L, 1) channel — total input (N, L, 1).
    def _th_df(scaled_df: pd.DataFrame) -> pd.DataFrame:
        return scaled_df[[target_col]].copy()

    th_context = copy.copy(context)
    th_context["pc_cols"]           = []          # empty → X has 0 feature channels
    th_context["train_scaled_df"]   = _th_df(train_scaled_df)
    th_context["val_scaled_df"]     = _th_df(val_scaled_df)
    th_context["test_scaled_df"]    = _th_df(test_scaled_df)
    th_context["_th_mode"]          = True
    return th_context


def run_train_and_evaluate(context: dict) -> dict:
    """LSTM tuning + final-refit lifecycle (P0-3B, P0-4, P1).

    Sweep phase (TUNING):
      For every (lookback, batch_size): fit on Train (max-available-data per
      lookback, D3), evaluate on the SAME 123 Val target dates for every
      lookback via cross-boundary windowing (P0-4: history context drawn from
      the tail of Train, targets confined to Val). Only a LIGHTWEIGHT record
      (metrics, best_epoch, sample counts, date ranges) is retained per
      candidate -- P1: no `_last_*`/last-iteration state, no TF model objects
      kept in RAM across the loop (each model is dereferenced + the Keras
      session cleared immediately after its metrics are extracted).

    Selection: lowest Val_RMSE (or explicit config override), from the
    lightweight records ONLY -- Test is never touched during tuning.

    Final refit (P0-3B):
      Instantiate a NEW model, refit on the unique Train+Val development set
      for exactly `best_epoch` epochs, no EarlyStopping/ReduceLROnPlateau.
      Forecast ALL 167 Test targets via cross-boundary windowing (context =
      tail(Train+Val, lookback)).
    """
    PROJECT_ROOT = context["PROJECT_ROOT"]
    lookback_values = context["lookback_values"]
    batch_size_values = context["batch_size_values"]
    epochs = context["epochs"]
    train_scaled_df = context["train_scaled_df"]
    val_scaled_df = context["val_scaled_df"]
    test_scaled_df = context["test_scaled_df"]
    x_scaler = context["x_scaler"]
    y_scaler = context["y_scaler"]
    pc_cols = context["pc_cols"]
    target_col = context["target_col"]

    # Load config for hyperparameters
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    lstm_cfg = config.get("lstm", {})
    preprocess_cfg = config.get("preprocess", {})

    # Get hyperparameters from config
    learning_rate = lstm_cfg.get("learning_rate", 1e-3)
    use_overlap_val = preprocess_cfg.get("use_overlap_val", False)
    base_patience = lstm_cfg.get("early_stopping_patience", 15)
    reduce_lr_patience = lstm_cfg.get("reduce_lr_patience", 7)
    reduce_lr_factor = lstm_cfg.get("reduce_lr_factor", 0.5)
    min_lr = lstm_cfg.get("min_lr", 1e-5)
    min_delta = lstm_cfg.get("min_delta", 1e-4)
    min_epochs = lstm_cfg.get("min_epochs", 20)
    overlap_multiplier = lstm_cfg.get("overlap_patience_multiplier", 2.0)

    # Architecture params
    lstm_units = lstm_cfg.get("lstm_units", [64, 32])
    dense_units = lstm_cfg.get("dense_units", [16])
    dropout_rate = lstm_cfg.get("dropout_rate", 0.2)
    use_batch_norm = lstm_cfg.get("use_batch_norm", False)

    def build_model(lookback: int, n_features: int):
        return _build_lstm_model(
            lookback, n_features,
            learning_rate=learning_rate, lstm_units=lstm_units,
            dense_units=dense_units, dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm, seed=42,
        )

    # Export target: read from config, fallback to auto (best Val_RMSE)
    _export_lookback = lstm_cfg.get("export_lookback", None)
    _export_batch_size = lstm_cfg.get("export_batch_size", None)

    results_dir = PROJECT_ROOT / "outputs" / "lstm_vnindex_sweep"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── TUNING PHASE ─────────────────────────────────────────────────────
    # `candidates` holds ONLY lightweight records (P1) -- no model/history
    # objects survive past the iteration that produced them.
    all_results = []
    candidates: dict = {}

    for lookback in lookback_values:
        # Train: MAXIMUM available train data per lookback (D3) -- targets
        # vary in count by lookback (n_train - lookback), NOT cross-boundary.
        X_train, y_train, train_dates = make_windowed_data(train_scaled_df, pc_cols, target_col, lookback)
        X_train_hist = add_target_history(train_scaled_df, target_col, lookback)
        X_train_final = np.concatenate([X_train, X_train_hist], axis=2)

        # Val: cross-boundary windowing (P0-4) -- context = tail(Train, lookback),
        # so ALL 123 Val dates are targets regardless of lookback.
        X_val, y_val, X_val_hist, val_dates = make_cross_boundary_windowed_data(
            train_scaled_df, val_scaled_df, pc_cols, target_col, lookback,
        )
        X_val_final = np.concatenate([X_val, X_val_hist], axis=2)

        print("=" * 90)
        print(f"LOOKBACK = {lookback}")
        print(f"Train samples: {len(X_train_final)} | Validation samples: {len(X_val_final)} (must be {len(val_scaled_df)} for every lookback)")
        assert len(X_val_final) == len(val_scaled_df), (
            f"P0-4 violation: Val samples ({len(X_val_final)}) != len(val_scaled_df) "
            f"({len(val_scaled_df)}) for lookback={lookback}."
        )

        for batch_size in batch_size_values:
            print("-" * 90)
            print(f"Training with batch_size = {batch_size}")

            model = build_model(lookback, X_train_final.shape[-1])

            patience_es = get_adaptive_patience(use_overlap_val, base_patience, overlap_multiplier)
            patience_rlr = get_adaptive_patience(use_overlap_val, reduce_lr_patience, overlap_multiplier)
            #Error EarlyStopping start at first epoch
            #es_start_epoch = min_epochs if use_overlap_val else 0
            es_start_epoch = min_epochs
            
            callbacks = [
                MinEpochEarlyStopping(
                    start_epoch=es_start_epoch,
                    monitor="val_loss",
                    patience=patience_es,
                    min_delta=min_delta,
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=reduce_lr_factor,
                    patience=patience_rlr,
                    min_lr=min_lr,
                ),
            ]

            print(f"[CONFIG] LR={learning_rate}, Patience_ES={patience_es}, Patience_RLR={patience_rlr}")
            if use_overlap_val:
                print(f"[OVERLAP] Validation overlap detected - patience increased by {overlap_multiplier}x")
                print(f"[OVERLAP] EarlyStopping hard-disabled until epoch {es_start_epoch} (min_epochs enforcement)")

            history = model.fit(
                X_train_final, y_train,
                validation_data=(X_val_final, y_val),
                epochs=epochs, batch_size=batch_size,
                callbacks=callbacks, verbose=1, shuffle=False,
            )

            train_pred = y_scaler.inverse_transform(model.predict(X_train_final, verbose=0)).ravel()
            val_pred = y_scaler.inverse_transform(model.predict(X_val_final, verbose=0)).ravel()
            y_train_true = y_scaler.inverse_transform(y_train).ravel()
            y_val_true = y_scaler.inverse_transform(y_val).ravel()

            train_rmse = rmse(y_train_true, train_pred)
            train_mae = float(mean_absolute_error(y_train_true, train_pred))
            train_mape = mape(y_train_true, train_pred)
            val_rmse = rmse(y_val_true, val_pred)
            val_mae = float(mean_absolute_error(y_val_true, val_pred))
            val_mape = mape(y_val_true, val_pred)

            print("Train  - RMSE:", f"{train_rmse:.5f}", "MAE:", f"{train_mae:.5f}", "MAPE:", f"{train_mape:.5f}%")
            print("Val    - RMSE:", f"{val_rmse:.5f}", "MAE:", f"{val_mae:.5f}", "MAPE:", f"{val_mape:.5f}%")

            try:
                best_epoch = int(np.argmin(history.history["val_loss"]) + 1) if "val_loss" in history.history else len(history.history.get("loss", []))
            except Exception:
                best_epoch = len(history.history.get("loss", []))

            if best_epoch < min_epochs:
                print(f"[WARN] Best epoch ({best_epoch}) < min_epochs ({min_epochs}) — "
                      f"possible early collapse; check tuning_history for val_loss trajectory.")

            if use_overlap_val and val_rmse > 0:
                overfit_ratio = train_rmse / val_rmse
                if overfit_ratio > 1.3:
                    print(f"[WARN] ALERT: Severe overfit detected (ratio={overfit_ratio:.2f})")
                elif overfit_ratio > 1.1:
                    print(f"[INFO]  Mild overfit detected (ratio={overfit_ratio:.2f})")

            row = {
                "Lookback": lookback, "Batch_size": batch_size,
                "Train_RMSE": train_rmse, "Train_MAE": train_mae, "Train_MAPE(%)": train_mape,
                "Val_RMSE": val_rmse, "Val_MAE": val_mae, "Val_MAPE(%)": val_mape,
                "Train_samples": len(X_train_final), "Val_samples": len(X_val_final),
                "Train_period_start": train_dates.min().strftime("%Y-%m-%d") if len(train_dates) > 0 else None,
                "Train_period_end": train_dates.max().strftime("%Y-%m-%d") if len(train_dates) > 0 else None,
                "Val_period_start": val_dates.min().strftime("%Y-%m-%d") if len(val_dates) > 0 else None,
                "Val_period_end": val_dates.max().strftime("%Y-%m-%d") if len(val_dates) > 0 else None,
                "Best_Epoch": best_epoch,
            }
            all_results.append(row)

            # ── P1: lightweight record ONLY. No model/history object kept. ──
            candidates[(lookback, batch_size)] = dict(row)
            candidates[(lookback, batch_size)]["val_dates"] = val_dates

            # ── Persist epoch-level tuning history BEFORE disposing ────────
            # (Phase E: never lose learning curves again; zero numerical impact)
            try:
                hist_dir = results_dir / "tuning_history"
                hist_dir.mkdir(parents=True, exist_ok=True)
                hist_df = pd.DataFrame(history.history)
                hist_df.index.name = "epoch"
                hist_df.index = hist_df.index + 1  # 1-indexed epochs
                hist_df.to_csv(hist_dir / f"tuning_history_LB{lookback}_BS{batch_size}.csv")
            except Exception as _hist_err:
                print(f"[WARN] Could not save tuning history: {_hist_err}")

            # Free the TF model/graph immediately — never accumulate 10
            # model objects in RAM across the sweep (P1).
            del model, history
            keras.backend.clear_session()

    summary_results = pd.DataFrame(all_results)
    summary_results.to_csv(results_dir / "sweep_summary.csv", index=False)
    print(f"\nSaved sweep summary to: {results_dir / 'sweep_summary.csv'}")
    print("\nSummary Results:")
    print(summary_results)

    # ── SELECTION (P0-3B): from lightweight records ONLY, Val_RMSE (or
    # explicit config override) -- Test has not been touched. ──────────────
    if _export_lookback is not None and _export_batch_size is not None:
        selected_lookback = int(_export_lookback)
        selected_batch_size = int(_export_batch_size)
        if (selected_lookback, selected_batch_size) not in candidates:
            raise KeyError(
                f"export_lookback/export_batch_size=({selected_lookback},{selected_batch_size}) "
                "is not a candidate produced by this sweep (check lookback_values/batch_size_values)."
            )
    else:
        if summary_results.empty:
            raise ValueError("LSTM sweep produced no candidates — cannot select a model.")
        best_idx = summary_results["Val_RMSE"].idxmin()
        selected_lookback = int(summary_results.loc[best_idx, "Lookback"])
        selected_batch_size = int(summary_results.loc[best_idx, "Batch_size"])

    selected_record = candidates[(selected_lookback, selected_batch_size)]
    best_epoch = int(selected_record["Best_Epoch"])
    print(f"\n[SELECT] Selected model by Val_RMSE: lookback={selected_lookback}, batch={selected_batch_size}"
          f"  Val_RMSE={selected_record['Val_RMSE']:.4f}  best_epoch={best_epoch}")

    # ── Copy selected candidate's tuning history as the reference ──────
    try:
        hist_src = results_dir / "tuning_history" / f"tuning_history_LB{selected_lookback}_BS{selected_batch_size}.csv"
        if hist_src.exists():
            import shutil
            shutil.copy2(hist_src, results_dir / "selected_tuning_history.csv")
            print(f"[SAVED] selected_tuning_history.csv (LB{selected_lookback}_BS{selected_batch_size})")
    except Exception:
        pass

    # ── FINAL REFIT (P0-3B, D4): NEW model instance, unique Train+Val,
    # epochs=best_epoch, NO EarlyStopping/ReduceLROnPlateau. Reuses the
    # shared final_refit_and_forecast() helper (also used by Phase 3D
    # multi-seed stability) so this logic is defined exactly once. ────────
    print(f"\n[FINAL REFIT] lookback={selected_lookback} batch={selected_batch_size} epochs={best_epoch} "
          f"on Train+Val — NO EarlyStopping / ReduceLROnPlateau")
    refit_result = final_refit_and_forecast(
        train_scaled_df=train_scaled_df, val_scaled_df=val_scaled_df, test_scaled_df=test_scaled_df,
        pc_cols=pc_cols, target_col=target_col, y_scaler=y_scaler,
        selected_lookback=selected_lookback, selected_batch_size=selected_batch_size,
        best_epoch=best_epoch, learning_rate=learning_rate, lstm_units=lstm_units,
        dense_units=dense_units, dropout_rate=dropout_rate, use_batch_norm=use_batch_norm,
        seed=42,
    )
    final_model = refit_result["model"]
    test_dates = refit_result["test_dates"]
    y_test_true = refit_result["y_test_true"]
    test_pred = refit_result["test_pred"]
    dev_dates = refit_result["dev_dates"]
    refit_metrics = refit_result["metrics"]
    X_dev_final = refit_result["X_dev_final"]

    print(f"[FINAL] Test — RMSE: {refit_metrics['Test_RMSE']:.5f}  MAE: {refit_metrics['Test_MAE']:.5f}  MAPE: {refit_metrics['Test_MAPE(%)']:.5f}%")
    print(f"[FINAL] Train+Val (dev) fit — RMSE: {refit_metrics['Dev_RMSE']:.5f}  MAE: {refit_metrics['Dev_MAE']:.5f}  MAPE: {refit_metrics['Dev_MAPE(%)']:.5f}%")

    # NOTE: Train_RMSE/Train_MAE/Train_MAPE(%) below (inherited from
    # `selected_record`) reflect the TUNING-phase model (fit on Train only,
    # per D3's max-available-data-per-lookback rule) -- they are retained
    # for audit purposes. Dev_RMSE/Dev_MAE/Dev_MAPE(%) reflect the actual
    # FINAL deployed model's fit quality on the Train+Val set it was
    # refit on (D4). Test_RMSE/etc. are this final model's genuine
    # out-of-sample performance.
    selected_metrics_row = dict(selected_record)
    selected_metrics_row.pop("val_dates", None)
    selected_metrics_row.update({
        **refit_metrics,
        "Test_samples": len(test_dates),
        "Test_period_start": test_dates.min().strftime("%Y-%m-%d") if len(test_dates) > 0 else None,
        "Test_period_end": test_dates.max().strftime("%Y-%m-%d") if len(test_dates) > 0 else None,
        "Dev_samples": len(X_dev_final),  # Train+Val development samples used for the final refit
        "Dev_period_start": dev_dates.min().strftime("%Y-%m-%d") if len(dev_dates) > 0 else None,
        "Dev_period_end": dev_dates.max().strftime("%Y-%m-%d") if len(dev_dates) > 0 else None,
        "Final_refit_epochs": best_epoch,
        "Final_refit_used_early_stopping": False,
    })

    pred_table = pd.DataFrame({
        "Date": test_dates,
        "Actual_VNINDEX": y_test_true,
        "Predicted_VNINDEX": test_pred,
        "Residual": y_test_true - test_pred,
    })
    selected_pred_filename = f"predictions_lookback_{selected_lookback}_batch_{selected_batch_size}.csv"
    pred_table.to_csv(results_dir / selected_pred_filename, index=False)
    print(f"Saved: {results_dir / selected_pred_filename}")

    context.update({
        "results_dir": results_dir,
        "summary_results": summary_results,
        "selected_lookback": selected_lookback,
        "selected_batch_size": selected_batch_size,
        "selected_train_dates": dev_dates,   # dates actually used to fit the FINAL model (Train+Val)
        "selected_val_dates": selected_record["val_dates"],  # the 123 Val dates used for selection
        "selected_test_dates": test_dates,   # the 167 Test dates forecast by the final model
        "selected_history": None,
        "selected_model": final_model,
        "selected_metrics_row": selected_metrics_row,
        "selected_pred_filename": selected_pred_filename,
        "X_train_final": X_dev_final,
        "final_refit_epochs": best_epoch,
        "final_refit_used_early_stopping": False,
    })
    return context
