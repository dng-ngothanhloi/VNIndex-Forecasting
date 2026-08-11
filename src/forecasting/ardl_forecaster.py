"""
ardl_forecaster.py – ARDLForecaster (src/forecasting)
============================================================
BaseForecaster-compliant adapter over the scientifically-corrected ARDL
pipeline (src/forecasting/ardl/{common,sweep}.py). This is the ONE
canonical ARDL implementation in this codebase -- there is no separate
"context dict" ARDL algorithm and "OOP wrapper" ARDL algorithm; this class
reuses `rolling_one_step_forecast` and `select_best_model` directly rather
than re-deriving the sweep/selection/rolling-forecast logic.

Scientific protocol preserved (P0-2/P0-3A/P0-X, forecasting-protocol-audit.md):
  - causal=True (no PC.L0), fixed hold_back across all (p,q) candidates.
  - Each candidate is fit on TRAIN ONLY; BIC/AIC/HQIC come from that fit.
  - RMSE_val/etc. come from a FIXED-MODEL rolling one-step-ahead forecast
    over ALL Val dates (never the model's own prior predictions).
  - Test is never touched during selection. After selection, ONE final
    ARDL is refit on unique Train+Val, and Test is forecast via the same
    rolling one-step-ahead helper.

Contract note: BaseForecaster.predict(X) normally receives only feature
columns, but genuine rolling one-step-ahead forecasting requires the
ACTUAL target history at t-1..t-max_lag for every target date -- there is
no other legitimate source for it (recursively substituting the model's
own prior predictions is exactly the leakage-adjacent behavior P0-X
forbids). Per the precedent already established by this codebase's
LSTMForecaster.predict() (which requires the target column in X for its
own target-history windowing), predict(X) here requires X to include the
target column too. This is disclosed, not silently assumed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.ardl import ARDL

from .ardl.common import (
    diagnostics_from_residuals,
    mape,
    paired_valid,
    rmse,
    rolling_one_step_forecast,
)
from .ardl.sweep import select_best_model

from .base import BaseForecaster, NotFittedError


class ARDLForecaster(BaseForecaster):
    """Wraps the corrected ARDL(P,Q)+PCA sweep-and-select-and-forecast
    pipeline (src/forecasting/ardl/sweep.py::run_sweep_ardl +
    src/forecasting/ardl/forecast.py::run_select_and_forecast) behind the
    BaseForecaster contract, reusing the same rolling-one-step-ahead
    helper and selection function rather than re-deriving them.

    Constructor arguments mirror configs/config.yaml's `ardl:` section
    keys exactly (Configuration Governance: no hardcoded hidden
    parameters) -- explicit constructor arguments, no file reading here.

    Public API is exactly: fit, predict, get_metadata.
    """

    def __init__(
        self,
        p_values: Optional[List[int]] = None,
        q_values: Optional[List[int]] = None,
        causal: bool = True,
        hold_back: int = 5,
        selection_criterion: str = "BIC",
        criterion_threshold: Optional[float] = None,
        rmse_val_threshold: Optional[float] = None,
        use_ensemble: bool = False,
        ensemble_top_n: int = 3,
        ensemble_criterion: str = "RMSE_val",
    ) -> None:
        self.p_values = p_values if p_values is not None else [1, 2, 3, 4, 5]
        self.q_values = q_values if q_values is not None else [1, 2, 3, 4, 5]
        self.causal = causal
        self.hold_back = hold_back
        self.selection_criterion = selection_criterion
        self.criterion_threshold = criterion_threshold
        self.rmse_val_threshold = rmse_val_threshold
        self.use_ensemble = use_ensemble
        self.ensemble_top_n = ensemble_top_n
        self.ensemble_criterion = ensemble_criterion

        self._is_fitted = False
        self._selected_pair: Optional[Tuple[int, int]] = None
        self._final_res = None
        self._sweep_table: Optional[pd.DataFrame] = None
        self._metrics: Optional[Dict[str, float]] = None
        self._diag: Optional[Dict[str, float]] = None
        self._y_trainval: Optional[pd.Series] = None
        self._X_trainval: Optional[pd.DataFrame] = None
        self._target_col: Optional[str] = None
        self._feature_columns: Optional[List[str]] = None
        self._ensemble_results: Optional[Dict[Tuple[int, int], Any]] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "ARDLForecaster":
        y_train = y_train.astype(float)
        X_train = X_train.astype(float)
        target_col = y_train.name if y_train.name is not None else "target"
        self._target_col = target_col
        self._feature_columns = list(X_train.columns)

        has_val = X_val is not None and y_val is not None
        if not has_val:
            raise ValueError(
                "ARDLForecaster.fit() requires X_val/y_val: the corrected "
                "protocol (P0-3A) selects (p,q) via a Train-only fit + "
                "rolling one-step-ahead forecast over Val, which needs an "
                "actual Val split -- there is no legitimate Train-only "
                "selection path under this protocol."
            )
        y_val = y_val.astype(float)
        X_val = X_val.astype(float)
        y_trainval = pd.concat([y_train, y_val], axis=0).sort_index()
        X_trainval = pd.concat([X_train, X_val], axis=0).sort_index()

        pq_pairs = [(p, q) for p in self.p_values for q in self.q_values]
        candidates: Dict[Tuple[int, int], dict] = {}
        sweep_rows = []

        for p, q in pq_pairs:
            row: Dict[str, Any] = {"P": p, "Q": q}
            try:
                model_train = ARDL(
                    endog=y_train, lags=p, exog=X_train, order=q,
                    causal=self.causal, trend="c", hold_back=self.hold_back,
                )
                res_train = model_train.fit()

                pred_val = rolling_one_step_forecast(res_train, y_trainval, X_trainval, y_val.index)
                y_v_eval, y_v_pred = paired_valid(y_val, pred_val)

                row.update({
                    "Status": "OK",
                    "Num_Params": int(len(res_train.params)),
                    "AIC": float(res_train.aic),
                    "BIC": float(res_train.bic),
                    "HQIC": float(res_train.hqic),
                    "RMSE_val": rmse(y_v_eval, y_v_pred),
                    "MAE_val": float(np.mean(np.abs(y_v_eval - y_v_pred))),
                    "MAPE_val(%)": mape(y_v_eval, y_v_pred),
                })
                candidates[(p, q)] = {"res_train": res_train}
            except Exception as exc:
                row.update({
                    "Status": f"FAIL: {type(exc).__name__}",
                    "Num_Params": np.nan, "AIC": np.nan, "BIC": np.nan, "HQIC": np.nan,
                    "RMSE_val": np.nan, "MAE_val": np.nan, "MAPE_val(%)": np.nan,
                })
            sweep_rows.append(row)

        sweep_table = pd.DataFrame(sweep_rows)

        # Reused UNMODIFIED from src/forecasting/ardl/sweep.py -- same
        # two-condition filter and lowest-criterion selection.
        selected_pair = select_best_model(
            sweep_table,
            criterion=self.selection_criterion,
            criterion_threshold=self.criterion_threshold,
            rmse_val_threshold=self.rmse_val_threshold,
        )
        if selected_pair not in candidates:
            raise KeyError(f"Selected pair {selected_pair} was not fit successfully in the sweep.")

        # ── ONE final refit on unique Train+Val (P0-3A) ────────────────
        p, q = selected_pair
        final_model = ARDL(
            endog=y_trainval, lags=p, exog=X_trainval, order=q,
            causal=self.causal, trend="c", hold_back=self.hold_back,
        )
        final_res = final_model.fit()

        y_trainval_eval, pred_trainval_eval = paired_valid(y_trainval, final_res.fittedvalues)
        metrics = {
            "RMSE_trainval": rmse(y_trainval_eval, pred_trainval_eval),
            "MAE_trainval": float(np.mean(np.abs(y_trainval_eval - pred_trainval_eval))),
            "MAPE_trainval(%)": mape(y_trainval_eval, pred_trainval_eval),
        }
        diag = diagnostics_from_residuals(final_res.resid)

        # Optional Top-N ensemble: each candidate refit on Train+Val too,
        # so the ensemble respects the same information boundary.
        ensemble_results = None
        if self.use_ensemble:
            sweep_ok = sweep_table[sweep_table["Status"] == "OK"].copy()
            crit = self.ensemble_criterion if self.ensemble_criterion in sweep_ok.columns else "RMSE_val"
            top_rows = sweep_ok.nsmallest(self.ensemble_top_n, crit)
            top_pairs = [(int(r["P"]), int(r["Q"])) for _, r in top_rows.iterrows() if (int(r["P"]), int(r["Q"])) in candidates]
            if len(top_pairs) >= 2:
                ensemble_results = {}
                for tp, tq in top_pairs:
                    ens_model = ARDL(
                        endog=y_trainval, lags=tp, exog=X_trainval, order=tq,
                        causal=self.causal, trend="c", hold_back=self.hold_back,
                    )
                    ensemble_results[(tp, tq)] = ens_model.fit()

        self._is_fitted = True
        self._selected_pair = selected_pair
        self._final_res = final_res
        self._sweep_table = sweep_table
        self._metrics = metrics
        self._diag = diag
        self._y_trainval = y_trainval
        self._X_trainval = X_trainval
        self._ensemble_results = ensemble_results
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError(
                "ARDLForecaster.predict() called before fit(). Call fit(X_train, y_train, X_val, y_val) first."
            )
        if self._target_col not in X.columns:
            raise KeyError(
                f"ARDLForecaster.predict(X): X must include the target column "
                f"'{self._target_col}' (rolling one-step-ahead forecasting "
                f"requires actual observed history, per P0-X -- see class docstring)."
            )

        y_full = pd.concat([self._y_trainval, X[self._target_col].astype(float)]).sort_index()
        X_full = pd.concat([self._X_trainval, X[self._feature_columns].astype(float)]).sort_index()

        pred = rolling_one_step_forecast(self._final_res, y_full, X_full, X.index)

        if self._ensemble_results:
            preds = [pred.values]
            for pair, res in self._ensemble_results.items():
                if pair == self._selected_pair:
                    continue
                preds.append(rolling_one_step_forecast(res, y_full, X_full, X.index).values)
            return np.mean(preds, axis=0)

        return pred.values

    def get_metadata(self) -> Dict[str, Any]:
        if not self._is_fitted:
            raise NotFittedError(
                "ARDLForecaster.get_metadata() called before fit(). Call fit(X_train, y_train, X_val, y_val) first."
            )
        return {
            "method": "ARDL",
            "selected_pair": self._selected_pair,
            "causal": self.causal,
            "hold_back": self.hold_back,
            "metrics": self._metrics,
            "diagnostics": self._diag,
            "sweep_table": self._sweep_table,
            "ensemble_pairs": list(self._ensemble_results.keys()) if self._ensemble_results else None,
            "feature_columns": self._feature_columns,
        }
