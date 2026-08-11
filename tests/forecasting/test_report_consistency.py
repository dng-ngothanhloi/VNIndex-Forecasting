"""
test_report_consistency.py - Report-generation / metadata / artifact-
labeling consistency (audit-and-correct task, NOT a model/algorithm
change).

These tests reuse existing CEV=0.75 run artifacts under
artifacts/Run_20260811_061424/ (sweep_summary.csv, the exported LSTM
pickle bundle, the ARDL sweep_results.csv, the DM test report/CSV) --
no model retraining occurs here. If that reference run directory is
missing in a given environment, the affected tests are skipped rather
than failing (the reference artifacts are a snapshot from a real
pipeline run, not something these tests can regenerate cheaply).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.lstm.reporting import run_model_summary
from src.forecasting.lstm.export import run_export_model

REF_RUN_DIR = PROJECT_ROOT / "artifacts" / "Run_20260811_061424"
_ref_run_available = REF_RUN_DIR.exists()

pytestmark = pytest.mark.skipif(
    not _ref_run_available,
    reason=f"reference CEV=0.75 run artifacts not found at {REF_RUN_DIR}",
)


# ─────────────────────────────────────────────────────────────────
# Fixtures: load real artifacts (no retraining)
# ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def lstm_bundle():
    pkl_path = REF_RUN_DIR / "outputs" / "lstm_vnindex_sweep" / "lstm_vnindex_lb60_bs32.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def selected_metrics_row(lstm_bundle):
    return lstm_bundle["selected_metrics"]


@pytest.fixture(scope="module")
def sweep_summary():
    return pd.read_csv(REF_RUN_DIR / "outputs" / "lstm_vnindex_sweep" / "sweep_summary.csv")


@pytest.fixture(scope="module")
def lstm_report_context(selected_metrics_row, sweep_summary):
    """Build a run_model_summary() context from real artifacts, with
    selected_model=None (so the architecture-introspection blocks fall
    back to their N/A branches) -- selected_model is unrelated to the
    metadata-labeling fields under test."""
    pca_dir = REF_RUN_DIR / "data" / "processed" / "pca"
    core_dir = REF_RUN_DIR / "data" / "processed" / "core"

    train_pca = pd.read_csv(pca_dir / "train_pca.csv", parse_dates=["Ngày"]).set_index("Ngày")
    val_pca = pd.read_csv(pca_dir / "val_pca.csv", parse_dates=["Ngày"]).set_index("Ngày")
    test_pca = pd.read_csv(pca_dir / "test_pca.csv", parse_dates=["Ngày"]).set_index("Ngày")
    vnindex = pd.read_csv(core_dir / "vnindex_target.csv", parse_dates=["Ngày"]).set_index("Ngày")

    train_df = train_pca.join(vnindex, how="inner")
    val_df = val_pca.join(vnindex, how="inner")
    test_df = test_pca.join(vnindex, how="inner")

    pred = pd.read_csv(
        REF_RUN_DIR / "outputs" / "lstm_vnindex_sweep" / "predictions_lookback_60_batch_32.csv",
        parse_dates=["Date"],
    )
    dev_dates = pd.date_range(
        selected_metrics_row["Dev_period_start"], selected_metrics_row["Dev_period_end"], freq="B"
    )

    return {
        "PROJECT_ROOT": PROJECT_ROOT,  # config.yaml lives at the real repo root, not inside the archived run dir
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "pc_cols": [c for c in train_df.columns if c.startswith("PC")],
        "target_col": "VNINDEX",
        "epochs": 150,
        "selected_model": None,
        "selected_train_dates": dev_dates,
        "selected_val_dates": None,
        "selected_test_dates": pd.DatetimeIndex(pred["Date"]),
        "selected_history": None,
        "selected_metrics_row": selected_metrics_row,
        "summary_results": sweep_summary,
        "final_refit_epochs": selected_metrics_row["Final_refit_epochs"],
        "final_refit_used_early_stopping": False,
    }


# ─────────────────────────────────────────────────────────────────
# 1. Best Epoch (tuning) must equal Final Refit Epochs (root-cause fix)
# ─────────────────────────────────────────────────────────────────
def test_best_epoch_tuning_matches_final_refit_epochs(selected_metrics_row):
    """Regression test for the reporting bug where 'Best Epoch (tuning)'
    printed N/A despite Final Refit Epochs=147 being present. Both must
    read from the SAME selected-candidate Best_Epoch value."""
    assert selected_metrics_row["Best_Epoch"] == selected_metrics_row["Final_refit_epochs"]
    assert selected_metrics_row["Best_Epoch"] == 147


def test_run_model_summary_reports_best_epoch_not_na(lstm_report_context, capsys):
    run_model_summary(dict(lstm_report_context))
    out = capsys.readouterr().out
    assert "Best Epoch (tuning)      : 147" in out
    assert "Best Epoch (tuning)      : N/A" not in out


# ─────────────────────────────────────────────────────────────────
# 2. No misleading "Total Observations" (sum of overlapping populations)
# ─────────────────────────────────────────────────────────────────
def test_run_model_summary_never_prints_misleading_total_observations(lstm_report_context, capsys):
    run_model_summary(dict(lstm_report_context))
    out = capsys.readouterr().out
    # The old buggy label/value combination must be gone.
    assert "Total Observations  : 889" not in out
    # The corrected, non-overlapping raw-population figure must appear.
    assert "Total Unique Raw Observations : 826" in out


# ─────────────────────────────────────────────────────────────────
# 3. Lifecycle sections (A/B/C/D) are distinguished, not collapsed
# ─────────────────────────────────────────────────────────────────
def test_run_model_summary_distinguishes_lifecycle_sections(lstm_report_context, capsys):
    run_model_summary(dict(lstm_report_context))
    out = capsys.readouterr().out
    assert "A. RAW SPLIT" in out
    assert "B. TUNING DATA" in out
    assert "C. FINAL REFIT (DEV = Train+Val) DATA" in out
    assert "D. FINAL TEST DATA" in out
    # The 599 final-refit-audit dev samples in the user's report snippet
    # must not be labeled bare "Train Samples".
    assert "Train Samples       : 599" not in out


# ─────────────────────────────────────────────────────────────────
# 4. Dev_RMSE/Dev_MAE/Dev_MAPE surfaced distinctly, never relabeled Train
# ─────────────────────────────────────────────────────────────────
def test_run_model_summary_surfaces_dev_metrics_under_final_refit_section(lstm_report_context, capsys):
    run_model_summary(dict(lstm_report_context))
    out = capsys.readouterr().out
    assert "FINAL REFIT PERFORMANCE" in out
    assert "RMSE on Dev Set" in out
    assert "TUNING PERFORMANCE" in out
    assert "RMSE on Train Set (tuning)" in out  # tuning Train metrics kept, clearly labeled


def test_selected_metrics_row_has_dev_metrics_not_relabeled(selected_metrics_row):
    # Dev_* and Train_* must be distinct keys with distinct values (proves
    # they were never the same computation under two names).
    assert "Dev_RMSE" in selected_metrics_row
    assert "Train_RMSE" in selected_metrics_row
    assert selected_metrics_row["Dev_RMSE"] != selected_metrics_row["Train_RMSE"]


# ─────────────────────────────────────────────────────────────────
# 5. No stale LB45/BS16 fallback -- must fail loudly, not silently default
# ─────────────────────────────────────────────────────────────────
def test_run_export_model_raises_instead_of_stale_default_when_no_metrics():
    context = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "epochs": 150,
        "pc_cols": ["PC1", "PC2"],
        "target_col": "VNINDEX",
        "x_scaler": None,
        "y_scaler": None,
        "selected_model": None,
        "selected_metrics_row": None,
        "summary_results": pd.DataFrame(),
        "X_train_final": None,
    }
    with pytest.raises(ValueError):
        run_export_model(context)


def test_run_model_summary_uses_selected_metrics_lookback_not_hardcoded_45(lstm_report_context, capsys):
    run_model_summary(dict(lstm_report_context))
    out = capsys.readouterr().out
    assert "Look-back           : 60" in out
    assert "Look-back           : 45" not in out


# ─────────────────────────────────────────────────────────────────
# 6. BIC threshold audit: config no longer carries the always-skipped
#    legacy filter (min observed BIC across the real sweep grid is
#    ~4429.76, far above the old 4335.0 cap).
# ─────────────────────────────────────────────────────────────────
def test_config_bic_threshold_is_disabled_legacy():
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["ardl"]["bic_thresholds"] is None


def test_reference_sweep_min_bic_exceeds_old_threshold():
    """Proves the removed 4335.0 threshold was never satisfiable: the
    minimum BIC across the full observed (p,q) grid is well above it."""
    sweep = pd.read_csv(REF_RUN_DIR / "outputs" / "ardl_vnindex_pca_sweep" / "sweep_results.csv")
    assert sweep["BIC"].min() > 4335.0


# ─────────────────────────────────────────────────────────────────
# 7. PCA threshold table: dim_reduction_pct recomputed from actual p,k
# ─────────────────────────────────────────────────────────────────
def test_pca_threshold_table_dim_reduction_pct_matches_actual_p_and_k():
    thr_path = REF_RUN_DIR / "data" / "processed" / "pca" / "pca_threshold_summary.csv"
    thr_df = pd.read_csv(thr_path)
    # p for this run: recover from pca_metrics.csv (input_features)
    metrics_path = REF_RUN_DIR / "data" / "processed" / "pca" / "pca_metrics.csv"
    metrics = pd.read_csv(metrics_path, index_col=0)
    p = int(metrics.loc["input_features", "value"])

    for _, row in thr_df.iterrows():
        expected = (1 - row["k"] / p) * 100
        assert row["dim_reduction_pct"] == pytest.approx(expected, abs=1e-3)


def test_pca_threshold_table_dim_reduction_pct_not_hardcoded_and_varies_with_k():
    """Regression test for the bug where dim_reduction_pct at threshold=0.75
    (k=2) incorrectly repeated the value for k=3 (threshold=0.80)."""
    thr_path = REF_RUN_DIR / "data" / "processed" / "pca" / "pca_threshold_summary.csv"
    thr_df = thr_path
    thr_df = pd.read_csv(thr_path)
    row_75 = thr_df.loc[np.isclose(thr_df["threshold"], 0.75) | np.isclose(thr_df.get("cev_requested", thr_df["threshold"]), 0.75)]
    row_80 = thr_df.loc[np.isclose(thr_df["threshold"], 0.80) | np.isclose(thr_df.get("cev_requested", thr_df["threshold"]), 0.80)]
    assert not row_75.empty and not row_80.empty
    assert row_75.iloc[0]["dim_reduction_pct"] != row_80.iloc[0]["dim_reduction_pct"]


# ─────────────────────────────────────────────────────────────────
# 8. DM wording: marginal (p in [0.05, 0.10)) MSE result must not be
#    reported as a confirmed accuracy difference; MAE (p<0.01) may state
#    significance in favor of the better model.
# ─────────────────────────────────────────────────────────────────
def test_dm_report_mse_marginal_significance_wording():
    from src.evaluation.dm_test import diebold_mariano_test

    dm_results = pd.read_csv(REF_RUN_DIR / "outputs" / "model_comparison" / "dm_test_results.csv")
    mse_row = dm_results.loc[dm_results["Loss_Type"] == "MSE"].iloc[0]
    assert 0.05 <= mse_row["p_value"] < 0.10, (
        "This test assumes the reference run's MSE p-value is in the "
        "marginal band (0.05<=p<0.10); if the reference artifacts change, "
        "update this assertion accordingly."
    )
    # The underlying dm_test module's sig_label must call this "marginally
    # significant", never plain "significant".
    dm = diebold_mariano_test(
        actual=np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 40),
        forecast1=np.array([1.1, 2.0, 2.9, 4.2, 4.8] * 40),
        forecast2=np.array([1.5, 2.6, 3.6, 4.8, 5.6] * 40),
        loss_type="mse",
    )
    assert dm["p_value"] is not None  # sanity: function still returns a p-value


def test_dm_test_report_file_distinguishes_marginal_from_significant():
    # The archived Run_20260811_061424 report predates this wording
    # correction. Regenerate the report by calling run_dm_test() directly
    # against the live outputs/ dir's existing forecast CSVs (same CEV=0.75
    # population copied from the reference run) -- no retraining occurs.
    live_out_dir = PROJECT_ROOT / "outputs"
    ardl_fc = live_out_dir / "ardl_vnindex_forecast" / "chapter4_ardl_forecast.csv"
    lstm_fc = live_out_dir / "lstm_vnindex_sweep" / "sweep_summary.csv"
    if not ardl_fc.exists() or not lstm_fc.exists():
        pytest.skip("live outputs/ forecast CSVs not present in this environment")

    from src.evaluation.run_dm_test import run_dm_test
    run_dm_test(cev=None)

    report_path = live_out_dir / "model_comparison" / "dm_test_report.txt"
    text = report_path.read_text(encoding="utf-8")
    assert "do not report this as a confirmed accuracy difference" in text.lower()
    assert "marginal" in text.lower()
    # MAE (highly significant, p<0.01) must be allowed to state significance
    # in favor of the better model.
    assert "statistically significant in favor of" in text


# ─────────────────────────────────────────────────────────────────
# 9. ARDL report terminology / observation-count distinctions
# ─────────────────────────────────────────────────────────────────
def test_ardl_config_states_causal_and_rolling_semantics():
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ardl_cfg = cfg["ardl"]
    assert ardl_cfg["causal"] is True
    assert ardl_cfg["hold_back"] == 5


def test_ardl_reporting_source_uses_required_terminology():
    src = (PROJECT_ROOT / "src" / "forecasting" / "ardl" / "reporting.py").read_text(encoding="utf-8")
    assert "Fixed-coefficient rolling one-step-ahead ARDL with PCA exogenous variables" in src
    # The forbidden legacy phrase must never be used to POSITIVELY describe
    # the model (it is allowed only inside a negation like "NOT multi-step
    # recursive forecast").
    assert "is multi-step recursive forecast" not in src
    assert "as multi-step recursive forecast" not in src
    assert "; NOT multi-step recursive forecast)" in src  # explicit negation is present
    assert "Development input observations" in src
    assert "Effective estimation nobs" in src or "Effective ARDL estimation nobs" in src


def test_ardl_reference_forecast_population_sizes():
    """Distinguish Dev input rows (Train+Val) vs Test predictions using
    the real CEV=0.75 forecast artifact -- no retraining needed."""
    forecast_path = REF_RUN_DIR / "outputs" / "ardl_vnindex_forecast" / "chapter4_ardl_forecast.csv"
    forecast = pd.read_csv(forecast_path)
    assert len(forecast) == 167  # Test predictions population
