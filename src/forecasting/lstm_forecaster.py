"""
lstm_forecaster.py – LSTMForecaster (src/forecasting)
============================================================
BaseForecaster-compliant adapter over the scientifically-corrected LSTM
pipeline (src/forecasting/lstm/{data,sweep}.py). This is the ONE
canonical LSTM implementation in this codebase -- it reuses
`make_cross_boundary_windowed_data`, `get_adaptive_patience`,
`MinEpochEarlyStopping`, and the model-building helper directly rather
than re-deriving the sweep/selection/final-refit logic.

Scientific protocol preserved (P0-3B/P0-4/P1, forecasting-protocol-audit.md):
  - TUNING: for every (lookback, batch_size), fit on Train (max available
    data per lookback), evaluate on the SAME Val target dates for every
    lookback via cross-boundary windowing (history from Train's tail).
    Only a lightweight record is kept per candidate -- no TF model objects
    accumulate across the sweep.
  - SELECTION: lowest Val_RMSE, from lightweight records only. Test is
    never touched during tuning.
  - FINAL REFIT: a NEW model instance is refit on the unique Train+Val
    development set for exactly `best_epoch` epochs, no
    EarlyStopping/ReduceLROnPlateau. Test is forecast via cross-boundary
    windowing (history from Train+Val's tail).

Contract note: predict(X) needs a target-history channel (the LSTM
architecture concatenates PC features with target history), so X must
include the target column plus the tail of the fitted development set as
context -- callers pass X as the target split's DataFrame (matching the
already-established LSTMForecaster.predict() convention prior to this
migration: X must include the target column).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .lstm.data import add_target_history, make_windowed_data, make_cross_boundary_windowed_data
from .lstm.sweep import MinEpochEarlyStopping, get_adaptive_patience

from .base import BaseForecaster, NotFittedError


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mape(y_true, y_pred) -> float:
    eps = 1e-8
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


class LSTMForecaster(BaseForecaster):
    """Wraps the corrected LSTM lookback x batch_size sweep-and-select
    and final-Train+Val-refit pipeline
    (src/forecasting/lstm/sweep.py::run_train_and_evaluate) behind the
    BaseForecaster contract.

    Constructor arguments mirror configs/config.yaml's `lstm:` section
    keys exactly (Configuration Governance) -- explicit constructor
    arguments, no file reading here.

    Single-seed only (this class does not implement Multi_Seed_Set
    execution): `seed` selects ONE fixed seed for weight initialization,
    matching the corrected pipeline's single deployed model per fit()
    call. Multi-seed stability (Phase 3D) is implemented at the
    experiment-script layer by calling fit()/predict() once per seed, not
    inside this class.

    Public API is exactly: fit, predict, get_metadata.
    """

    def __init__(
        self,
        lookback_values: Optional[List[int]] = None,
        batch_size_values: Optional[List[int]] = None,
        epochs: int = 150,
        learning_rate: float = 1e-4,
        early_stopping_patience: int = 25,
        reduce_lr_patience: int = 10,
        reduce_lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        min_delta: float = 1e-4,
        overlap_patience_multiplier: float = 2.0,
        min_epochs: int = 30,
        lstm_units: Sequence[int] = (64, 32),
        dense_units: Sequence[int] = (16,),
        dropout_rate: float = 0.2,
        use_batch_norm: bool = False,
        use_overlap_val: bool = False,
        export_lookback: Optional[int] = None,
        export_batch_size: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.lookback_values = lookback_values if lookback_values is not None else [20, 30, 40, 50, 60]
        self.batch_size_values = batch_size_values if batch_size_values is not None else [16, 32]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.overlap_patience_multiplier = overlap_patience_multiplier
        self.min_epochs = min_epochs
        self.lstm_units = list(lstm_units)
        self.dense_units = list(dense_units)
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_overlap_val = use_overlap_val
        self.export_lookback = export_lookback
        self.export_batch_size = export_batch_size
        self.seed = seed

        self._is_fitted = False
        self._selected_lookback: Optional[int] = None
        self._selected_batch_size: Optional[int] = None
        self._selected_model = None
        self._x_scaler: Optional[StandardScaler] = None
        self._y_scaler: Optional[StandardScaler] = None
        self._sweep_table: Optional[pd.DataFrame] = None
        self._selected_metrics_row: Optional[dict] = None
        self._pc_cols: Optional[List[str]] = None
        self._target_col: Optional[str] = None
        self._train_scaled_df: Optional[pd.DataFrame] = None
        self._val_scaled_df: Optional[pd.DataFrame] = None
        self._final_refit_epochs: Optional[int] = None

    def _build_model(self, lookback: int, n_features: int):
        try:
            tf.keras.utils.set_random_seed(self.seed)
        except Exception:
            tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        layer_list = [layers.Input(shape=(lookback, n_features))]
        layer_list.append(layers.LSTM(self.lstm_units[0], return_sequences=True))
        if self.use_batch_norm:
            layer_list.append(layers.BatchNormalization())
        layer_list.append(layers.Dropout(self.dropout_rate))
        layer_list.append(layers.LSTM(self.lstm_units[1]))
        if self.use_batch_norm:
            layer_list.append(layers.BatchNormalization())
        layer_list.append(layers.Dropout(self.dropout_rate))
        layer_list.append(layers.Dense(self.dense_units[0], activation="relu"))
        layer_list.append(layers.Dense(1))

        model = keras.Sequential(layer_list)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
        )
        return model

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LSTMForecaster":
        has_val = X_val is not None and y_val is not None
        if not has_val:
            raise ValueError(
                "LSTMForecaster.fit() requires X_val/y_val: the corrected "
                "protocol (P0-3B) selects (lookback, batch_size) via "
                "cross-boundary Val evaluation, which needs an actual Val "
                "split -- there is no legitimate Train-only selection "
                "path under this protocol."
            )

        pc_cols = list(X_train.columns)
        target_col = y_train.name if y_train.name is not None else "target"
        self._pc_cols = pc_cols
        self._target_col = target_col

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_scaler.fit(X_train.astype(float))
        y_scaler.fit(y_train.astype(float).to_frame(target_col))

        def _scale(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
            scaled = pd.DataFrame(x_scaler.transform(X.astype(float)), index=X.index, columns=pc_cols)
            scaled[target_col] = y_scaler.transform(y.astype(float).to_frame(target_col)).ravel()
            return scaled

        train_scaled_df = _scale(X_train, y_train)
        val_scaled_df = _scale(X_val, y_val)

        all_results = []
        candidates: Dict[Tuple[int, int], dict] = {}

        for lookback in self.lookback_values:
            X_tr, y_tr, _ = make_windowed_data(train_scaled_df, pc_cols, target_col, lookback)
            X_tr_hist = add_target_history(train_scaled_df, target_col, lookback)
            X_tr_final = np.concatenate([X_tr, X_tr_hist], axis=2)

            X_v, y_v, X_v_hist, val_dates = make_cross_boundary_windowed_data(
                train_scaled_df, val_scaled_df, pc_cols, target_col, lookback,
            )
            X_v_final = np.concatenate([X_v, X_v_hist], axis=2)

            for batch_size in self.batch_size_values:
                model = self._build_model(lookback, X_tr_final.shape[-1])

                patience_es = get_adaptive_patience(self.use_overlap_val, self.early_stopping_patience, self.overlap_patience_multiplier)
                patience_rlr = get_adaptive_patience(self.use_overlap_val, self.reduce_lr_patience, self.overlap_patience_multiplier)
                es_start_epoch = self.min_epochs if self.use_overlap_val else 0

                callbacks = [
                    MinEpochEarlyStopping(
                        start_epoch=es_start_epoch, monitor="val_loss",
                        patience=patience_es, min_delta=self.min_delta,
                        restore_best_weights=True,
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", factor=self.reduce_lr_factor,
                        patience=patience_rlr, min_lr=self.min_lr,
                    ),
                ]

                history = model.fit(
                    X_tr_final, y_tr, validation_data=(X_v_final, y_v),
                    epochs=self.epochs, batch_size=batch_size,
                    callbacks=callbacks, verbose=0, shuffle=False,
                )

                train_pred = y_scaler.inverse_transform(model.predict(X_tr_final, verbose=0)).ravel()
                y_train_true = y_scaler.inverse_transform(y_tr).ravel()
                val_pred = y_scaler.inverse_transform(model.predict(X_v_final, verbose=0)).ravel()
                y_val_true = y_scaler.inverse_transform(y_v).ravel()

                try:
                    best_epoch = int(np.argmin(history.history["val_loss"]) + 1) if "val_loss" in history.history else len(history.history.get("loss", []))
                except Exception:
                    best_epoch = len(history.history.get("loss", []))

                row: Dict[str, Any] = {
                    "Lookback": lookback, "Batch_size": batch_size,
                    "Train_RMSE": _rmse(y_train_true, train_pred),
                    "Train_MAE": float(mean_absolute_error(y_train_true, train_pred)),
                    "Train_MAPE(%)": _mape(y_train_true, train_pred),
                    "Val_RMSE": _rmse(y_val_true, val_pred),
                    "Val_MAE": float(mean_absolute_error(y_val_true, val_pred)),
                    "Val_MAPE(%)": _mape(y_val_true, val_pred),
                    "Best_Epoch": best_epoch,
                }
                all_results.append(row)
                # Lightweight record ONLY (P1): no model/history object kept
                # past this iteration.
                candidates[(lookback, batch_size)] = dict(row)

                del model, history
                keras.backend.clear_session()

        sweep_table = pd.DataFrame(all_results)

        if self.export_lookback is not None and self.export_batch_size is not None:
            selected_lookback, selected_batch_size = int(self.export_lookback), int(self.export_batch_size)
        else:
            if sweep_table.empty:
                raise ValueError("LSTM sweep produced no candidates -- cannot select a model.")
            best_idx = sweep_table["Val_RMSE"].idxmin()
            selected_lookback = int(sweep_table.loc[best_idx, "Lookback"])
            selected_batch_size = int(sweep_table.loc[best_idx, "Batch_size"])

        selected_record = candidates[(selected_lookback, selected_batch_size)]
        best_epoch = int(selected_record["Best_Epoch"])

        # ── FINAL REFIT: NEW model, unique Train+Val, epochs=best_epoch,
        # no EarlyStopping/ReduceLROnPlateau (P0-3B/D4). ────────────────
        trainval_scaled_df = pd.concat([train_scaled_df, val_scaled_df], axis=0).sort_index()
        X_dev, y_dev, _ = make_windowed_data(trainval_scaled_df, pc_cols, target_col, selected_lookback)
        X_dev_hist = add_target_history(trainval_scaled_df, target_col, selected_lookback)
        X_dev_final = np.concatenate([X_dev, X_dev_hist], axis=2)

        final_model = self._build_model(selected_lookback, X_dev_final.shape[-1])
        final_model.fit(
            X_dev_final, y_dev, epochs=best_epoch, batch_size=selected_batch_size,
            verbose=0, shuffle=False,
        )

        self._is_fitted = True
        self._sweep_table = sweep_table
        self._x_scaler = x_scaler
        self._y_scaler = y_scaler
        self._selected_lookback = selected_lookback
        self._selected_batch_size = selected_batch_size
        self._selected_model = final_model
        self._selected_metrics_row = selected_record
        self._train_scaled_df = train_scaled_df
        self._val_scaled_df = val_scaled_df
        self._final_refit_epochs = best_epoch
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError(
                "LSTMForecaster.predict() called before fit(). Call fit(X_train, y_train, X_val, y_val) first."
            )
        if self._target_col not in X.columns:
            raise KeyError(
                f"LSTMForecaster.predict(X): X must include the target column "
                f"'{self._target_col}' (needed for target-history windowing "
                f"and cross-boundary context, see class docstring)."
            )

        lookback = self._selected_lookback
        scaled = pd.DataFrame(
            self._x_scaler.transform(X[self._pc_cols].astype(float)), index=X.index, columns=self._pc_cols
        )
        scaled[self._target_col] = self._y_scaler.transform(X[[self._target_col]].astype(float)).ravel()

        trainval_scaled_df = pd.concat([self._train_scaled_df, self._val_scaled_df], axis=0).sort_index()
        X_win, _, X_hist, _ = make_cross_boundary_windowed_data(
            trainval_scaled_df, scaled, self._pc_cols, self._target_col, lookback,
        )
        X_final = np.concatenate([X_win, X_hist], axis=2)

        pred_scaled = self._selected_model.predict(X_final, verbose=0)
        return self._y_scaler.inverse_transform(pred_scaled).ravel()

    def get_metadata(self) -> Dict[str, Any]:
        if not self._is_fitted:
            raise NotFittedError(
                "LSTMForecaster.get_metadata() called before fit(). Call fit(X_train, y_train, X_val, y_val) first."
            )
        return {
            "method": "LSTM",
            "selected_lookback": self._selected_lookback,
            "selected_batch_size": self._selected_batch_size,
            "final_refit_epochs": self._final_refit_epochs,
            "sweep_table": self._sweep_table,
            "selected_metrics_row": self._selected_metrics_row,
            "seed": self.seed,
        }
