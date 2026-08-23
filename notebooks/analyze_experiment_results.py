"""
analyze_experiment_results.py — Reusable analysis/report generator for a
representation-sweep run produced by `experiments/run_experiment.py --full-sweep`.

Purpose
-------
Turn the raw artifacts of one parent Run_* directory into the tables and
figures needed to answer O1-O4 / RQ1-RQ3 for a Q2 journal submission.

Design constraints (per project governance)
-------------------------------------------
* Read-only with respect to `artifacts/` — never mutates experiment outputs.
* Everything is derived from ONE parent run directory so the analysis is
  reproducible and auditable from a single artifact root.
* Deterministic output paths; no hidden state; explicit errors on missing
  required artifacts rather than silent fallback.
* Test-set predictions are used for REPORTING only. No model selection,
  no metric replacement, no seed cherry-picking happens here.
* Statistical tests are limited to the governed set: Diebold-Mariano and
  paired Wilcoxon signed-rank (Holm correction reported alongside raw p).

Usage
-----
    python notebooks/analyze_experiment_results.py \
        --run-dir artifacts/Run_20260823_130205

    # explicit output root (default: notebooks/analysis_output/<run_id>/)
    python notebooks/analyze_experiment_results.py \
        --run-dir artifacts/Run_20260823_130205 \
        --out-dir notebooks/analysis_output/custom

Outputs
-------
    <out-dir>/tables/*.csv      machine-readable tables
    <out-dir>/tables/*.md       same tables, markdown (paste-ready)
    <out-dir>/figures/*.png     comparison figures
    <out-dir>/analysis_data.json  consolidated numbers used in the report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.dm_test import diebold_mariano_test  # noqa: E402
from scipy import stats  # noqa: E402

# ── Presentation constants ───────────────────────────────────────────────────
CEV_ORDER = [0.75, 0.80, 0.85, 0.90, 0.95]
PCA_LABELS = [f"pca_cev_{c:.2f}" for c in CEV_ORDER]
NO_DR_LABEL = "no_dr"
ALL_LABELS = [NO_DR_LABEL] + PCA_LABELS

# Labels whose LSTM training must be inspected for premature-stop collapse.
# A run is flagged when the selected model's Best_Epoch is this small: with
# `restore_best_weights=True` the reported model is then effectively untrained.
COLLAPSE_EPOCH_THRESHOLD = 5

# Color-vision-deficiency-safe palette (Okabe & Ito, 2008). The previous
# palette used muted grays/browns (#7f7f7f vs #bfbfbf, #c0504d reused for
# every representation) that collapse into each other in grayscale print and
# are hard to tell apart for colorblind readers -- exactly the complaint this
# revision fixes. Roles are saturated and mutually distinguishable by hue
# alone, not just lightness.
PALETTE = {
    "ardl": "#0072B2",          # blue
    "lstm": "#D55E00",          # vermillion
    "persistence": "#333333",   # near-black (was a gray indistinguishable from ar1)
    "ar1": "#56B4E9",           # sky blue (was a near-identical light gray)
    "accent": "#009E73",        # bluish green
    "warn": "#CC79A7",          # reddish purple (shock / flag highlight)
    "neutral": "#999999",       # background / non-focal series (no-DR)
}

# Every representation gets ONE fixed colour + marker used consistently
# across every figure, so a reader does not have to relearn the mapping from
# figure to figure. Warm hues (vermillion/orange) are reserved for the two
# representations whose LSTM tuning grid is corrupted (Section 2b), so
# training instability is visually flagged everywhere they appear.
REP_STYLE = {
    "no_dr":        {"color": "#999999", "marker": "X"},
    "pca_cev_0.75": {"color": "#0072B2", "marker": "o"},   # valid
    "pca_cev_0.80": {"color": "#D55E00", "marker": "s"},   # corrupted (100% collapse)
    "pca_cev_0.85": {"color": "#E69F00", "marker": "^"},   # partially corrupted (70%)
    "pca_cev_0.90": {"color": "#009E73", "marker": "D"},   # valid
    "pca_cev_0.95": {"color": "#CC79A7", "marker": "v"},   # valid
}


def _rep_color(label: str) -> str:
    return REP_STYLE.get(label, {"color": "#333333"})["color"]


# ── Small helpers ────────────────────────────────────────────────────────────

def _hdr(text: str, width: int = 78) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _require(path: Path, what: str) -> Path:
    """Fail loudly on a missing required artifact (no silent fallback)."""
    if not path.exists():
        raise FileNotFoundError(f"Required {what} not found: {path}")
    return path


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_table(df: pd.DataFrame, out_dir: Path, name: str, *,
                caption: str = "", float_fmt: str = "%.4f",
                index: bool = False) -> None:
    """Persist one table as both CSV and markdown."""
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / f"{name}.csv", index=index, float_format=float_fmt)
    md = df.to_markdown(index=index, floatfmt=".4f")
    header = f"**{caption}**\n\n" if caption else ""
    (tables / f"{name}.md").write_text(header + md + "\n", encoding="utf-8")
    print(f"  [TABLE] {name}.csv / .md   ({len(df)} rows)")


def _save_fig(fig, out_dir: Path, name: str) -> None:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIGURE] {name}.png")


def holm_correction(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def wilcoxon_abs_error(actual, f1, f2) -> dict:
    """Paired Wilcoxon signed-rank on |error| differences (f1 vs f2).

    Negative median difference => f1 has the smaller absolute errors.
    """
    a = np.asarray(actual, dtype=float)
    d = np.abs(a - np.asarray(f1, dtype=float)) - np.abs(a - np.asarray(f2, dtype=float))
    nonzero = d[d != 0]
    if len(nonzero) < 10:
        return {"w_stat": np.nan, "p_value": np.nan,
                "median_abs_err_diff": float(np.median(d)), "n_nonzero": int(len(nonzero))}
    res = stats.wilcoxon(nonzero, alternative="two-sided")
    return {"w_stat": float(res.statistic), "p_value": float(res.pvalue),
            "median_abs_err_diff": float(np.median(d)), "n_nonzero": int(len(nonzero))}


# ── Artifact loaders ─────────────────────────────────────────────────────────

def _reference_child(run_dir: Path) -> Path:
    """A child whose preprocessing snapshot represents the shared protocol.

    Preprocessing is re-executed identically for every child, so any child's
    snapshot documents the shared Train/Val/Test protocol. We prefer a PCA
    child because it also carries the full PCA artifact set.
    """
    for label in PCA_LABELS + [NO_DR_LABEL]:
        cand = run_dir / label / "data" / "processed"
        if cand.exists():
            return run_dir / label
    raise FileNotFoundError(f"No child with a data/processed snapshot under {run_dir}")


def load_preprocessing(run_dir: Path) -> dict:
    """Section 0 inputs: cleaning, quality, splits, stationarity."""
    child = _reference_child(run_dir)
    dp = child / "data" / "processed"

    cleaned = pd.read_csv(_require(dp / "core" / "cleaned_data.csv", "cleaned_data"))
    target = pd.read_csv(_require(dp / "core" / "vnindex_target.csv", "vnindex_target"),
                         parse_dates=["Ngày"])
    valid = pd.read_csv(_require(dp / "core" / "valid_stocks.csv", "valid_stocks"))
    removed = pd.read_csv(_require(dp / "core" / "removed_stocks.csv", "removed_stocks"))
    splits = pd.read_csv(_require(dp / "splits" / "split_summary.csv", "split_summary"))
    corr = pd.read_csv(_require(dp / "quality" / "corr_summary.csv", "corr_summary"))
    missing = pd.read_csv(_require(dp / "quality" / "missing_dist.csv", "missing_dist"))
    outliers = pd.read_csv(_require(dp / "quality" / "outlier_log.csv", "outlier_log"))
    stat = pd.read_csv(_require(dp / "stationarity" / "stationarity_results.csv", "stationarity"))
    statlr = pd.read_csv(_require(dp / "stationarity" / "stationarity_logreturn.csv",
                                  "stationarity_logreturn"))

    scaled = pd.read_csv(dp / "splits" / "train_scaled.csv", nrows=2)

    return {
        "source_child": child.name,
        "n_obs": int(len(cleaned)),
        "n_features_cleaned": int(cleaned.shape[1] - 1),
        "n_valid_stocks": int(len(valid)),
        "n_removed_stocks": int(len(removed)),
        "n_scaled_cols": int(scaled.shape[1] - 1),
        "target_start": str(target["Ngày"].min().date()),
        "target_end": str(target["Ngày"].max().date()),
        "target_min": float(target["VNINDEX"].min()),
        "target_max": float(target["VNINDEX"].max()),
        "target_mean": float(target["VNINDEX"].mean()),
        "target_std": float(target["VNINDEX"].std()),
        "splits": splits,
        "corr_summary": corr,
        "missing_dist": missing,
        "n_outlier_symbols": int(len(outliers)),
        "total_outliers": int(outliers["n_outliers"].sum()),
        "stationarity": stat,
        "stationarity_logreturn": statlr,
        "target": target,
    }


def load_pca_structure(run_dir: Path) -> dict:
    """Section 1 inputs: per-CEV dimensionality + the shared PC spectrum."""
    rows = []
    for cev, label in zip(CEV_ORDER, PCA_LABELS):
        m = pd.read_csv(
            _require(run_dir / label / "data" / "processed" / "pca" / "pca_metrics.csv",
                     f"pca_metrics for {label}"),
            index_col=0)["value"]
        rows.append({
            "label": label,
            "CEV_requested": float(m["cev_threshold"]),
            "k": int(m["k_optimal"]),
            "CEV_achieved": float(m["cev_achieved"]),
            "p_original": int(m["input_features"]),
            "dim_reduction_pct": float(m["dim_reduction_pct"]),
            "max_pc_cross_corr": float(m["max_pc_cross_corr"]),
        })
    dim = pd.DataFrame(rows)

    # Governance check: the child must be fitted at the CEV its label claims.
    for _, r in dim.iterrows():
        expected = float(r["label"].split("_")[-1])
        if abs(r["CEV_requested"] - expected) > 1e-9:
            raise AssertionError(
                f"Representation identity violation: {r['label']} was fitted at "
                f"CEV={r['CEV_requested']}. Sweep results are not comparable.")

    # Widest child carries the deepest retained spectrum.
    widest = PCA_LABELS[int(dim["k"].idxmax())]
    spectrum = pd.read_csv(
        _require(run_dir / widest / "data" / "processed" / "pca" / "pca_eigenvalues.csv",
                 "pca_eigenvalues"))
    threshold_table = pd.read_csv(
        _require(run_dir / widest / "data" / "processed" / "pca" / "pca_threshold_summary.csv",
                 "pca_threshold_summary"))

    # PC1..PC6 loading structure — read from the child that retains >= 6 PCs.
    six = dim[dim["k"] >= 6]
    loadings = None
    if not six.empty:
        src = six.iloc[0]["label"]
        loadings = pd.read_csv(
            run_dir / src / "data" / "processed" / "pca" / "pca_loadings.csv", index_col=0)

    return {"dimensionality": dim, "spectrum": spectrum,
            "threshold_table": threshold_table, "loadings": loadings,
            "spectrum_source": widest}


def load_forecast_results(run_dir: Path) -> pd.DataFrame:
    """Section 2 inputs: per-child ARDL/LSTM/DM/multi-seed summary."""
    rows = []
    for label in ALL_LABELS:
        summary_path = run_dir / label / "results" / "run_summary.json"
        if not summary_path.exists():
            print(f"  [WARN] missing run_summary for {label}; skipped")
            continue
        s = _read_json(summary_path)
        repr_info = s.get("representation", {})
        ardl, lstm = s.get("ardl", {}), s.get("lstm", {})
        dm, ms = s.get("dm", {}), s.get("multiseed", {})
        rows.append({
            "label": label,
            "representation": repr_info.get("method"),
            "CEV_requested": repr_info.get("cev_requested"),
            "k": repr_info.get("k_actual"),
            "ARDL_P": ardl.get("ARDL_P"), "ARDL_Q": ardl.get("ARDL_Q"),
            "ARDL_RMSE": ardl.get("ARDL_RMSE"), "ARDL_MAE": ardl.get("ARDL_MAE"),
            "ARDL_MAPE": ardl.get("ARDL_MAPE"), "ARDL_R2": ardl.get("ARDL_R2"),
            "LSTM_LB": lstm.get("LSTM_LB"), "LSTM_BS": lstm.get("LSTM_BS"),
            "LSTM_best_epoch": lstm.get("LSTM_best_epoch"),
            "LSTM_RMSE": lstm.get("LSTM_RMSE"), "LSTM_MAE": lstm.get("LSTM_MAE"),
            "LSTM_MAPE": lstm.get("LSTM_MAPE"), "LSTM_R2": lstm.get("LSTM_R2"),
            "DM_MSE_p": dm.get("DM_MSE_p"), "DM_MAE_p": dm.get("DM_MAE_p"),
            "MS_RMSE_mean": ms.get("MS_RMSE_mean"), "MS_RMSE_std": ms.get("MS_RMSE_std"),
            "MS_MAE_mean": ms.get("MS_MAE_mean"), "MS_MAE_std": ms.get("MS_MAE_std"),
            "MS_R2_mean": ms.get("MS_R2_mean"), "MS_R2_std": ms.get("MS_R2_std"),
            "N_test": ardl.get("ARDL_N"),
        })
    if not rows:
        raise FileNotFoundError(f"No child run_summary.json found under {run_dir}")
    return pd.DataFrame(rows)


def load_lstm_training_diagnostics(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """LSTM sweep grids + epoch-level curves, to audit training stability.

    Returns (grid, curves) where `grid` has one row per (label, lookback,
    batch_size) and `curves` maps label -> {config -> val_loss series}.
    """
    grid_rows, curves = [], {}
    for label in ALL_LABELS:
        sweep_path = run_dir / label / "outputs" / "lstm_vnindex_sweep" / "sweep_summary.csv"
        if not sweep_path.exists():
            continue
        sweep = pd.read_csv(sweep_path)
        sweep.insert(0, "label", label)
        sweep["collapsed"] = sweep["Best_Epoch"] <= COLLAPSE_EPOCH_THRESHOLD
        grid_rows.append(sweep)

        hist_dir = run_dir / label / "outputs" / "lstm_vnindex_sweep" / "tuning_history"
        if hist_dir.exists():
            per_label = {}
            for f in sorted(hist_dir.glob("tuning_history_*.csv")):
                h = pd.read_csv(f)
                if "val_loss" in h.columns:
                    cfg = f.stem.replace("tuning_history_", "")
                    per_label[cfg] = h["val_loss"].to_numpy()
            curves[label] = per_label
    if not grid_rows:
        raise FileNotFoundError(f"No LSTM sweep_summary.csv found under {run_dir}")
    return pd.concat(grid_rows, ignore_index=True), curves


def load_prediction_series(run_dir: Path) -> dict:
    """Aligned Test-set prediction series for every model under comparison."""
    series: dict[str, pd.DataFrame] = {}

    for name, rel in [("Persistence", "baselines/persistence/predictions_test.csv"),
                      ("AR1", "baselines/ar1/predictions_test.csv")]:
        p = run_dir / rel
        if p.exists():
            d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")
            series[name] = d[["Actual_VNINDEX", "Predicted_VNINDEX"]]

    for label in ALL_LABELS:
        rpt = run_dir / label / "outputs" / "ardl_vnindex_report"
        hits = sorted(rpt.glob("predicted_vs_actual_test_P*_Q*.csv")) if rpt.exists() else []
        if hits:
            d = pd.read_csv(hits[0], parse_dates=["Date"]).set_index("Date")
            series[f"ARDL::{label}"] = d[["Actual_VNINDEX", "Predicted_VNINDEX"]]

        lsd = run_dir / label / "outputs" / "lstm_vnindex_sweep"
        hits = sorted(lsd.glob("predictions_lookback_*_batch_*.csv")) if lsd.exists() else []
        if hits:
            d = pd.read_csv(hits[0], parse_dates=["Date"]).set_index("Date")
            series[f"LSTM::{label}"] = d[["Actual_VNINDEX", "Predicted_VNINDEX"]]

    if not series:
        raise FileNotFoundError(f"No prediction series found under {run_dir}")
    return series


def load_multiseed(run_dir: Path) -> pd.DataFrame:
    """Per-seed Test metrics for every representation."""
    frames = []
    for label in ALL_LABELS:
        p = run_dir / label / "outputs" / "lstm_vnindex_multiseed" / "multiseed_test_metrics.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d.insert(0, "label", label)
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"No multiseed_test_metrics.csv found under {run_dir}")
    return pd.concat(frames, ignore_index=True)


def load_architecture_record(run_dir: Path, labels: list[str]) -> pd.DataFrame:
    """Section 6: exact trained configuration for the requested labels."""
    cfg_path = _require(PROJECT_ROOT / "configs" / "config.yaml", "config.yaml")
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    lstm_cfg, ardl_cfg = cfg.get("lstm", {}), cfg.get("ardl", {})

    rows = []
    for label in labels:
        meta = _read_json(_require(
            run_dir / label / "outputs" / "ardl_vnindex_report" / "ardl_meta.json",
            f"ardl_meta for {label}"))
        split_metrics = _read_json(_require(
            run_dir / label / "outputs" / "ardl_vnindex_report" / "metrics_by_split.json",
            f"metrics_by_split for {label}"))
        diag = _read_json(
            run_dir / label / "outputs" / "ardl_vnindex_report" / "ardl_diagnostics.json")
        summary = _read_json(run_dir / label / "results" / "run_summary.json")
        sweep = pd.read_csv(
            run_dir / label / "outputs" / "lstm_vnindex_sweep" / "sweep_summary.csv")
        sel = sweep.loc[sweep["Val_RMSE"].idxmin()]
        k = summary.get("representation", {}).get("k_actual")

        rows.append({
            "label": label,
            "k": k,
            # ── ARDL ──
            "ARDL_order_P": summary.get("ardl", {}).get("ARDL_P"),
            "ARDL_order_Q": summary.get("ardl", {}).get("ARDL_Q"),
            "ARDL_causal": ardl_cfg.get("causal"),
            "ARDL_hold_back": meta.get("hold_back"),
            "ARDL_num_params": meta.get("num_params"),
            "ARDL_nobs": meta.get("nobs"),
            "ARDL_selection": ardl_cfg.get("selection_criterion"),
            "ARDL_AIC": meta.get("aic"), "ARDL_BIC": meta.get("bic"),
            "ARDL_HQIC": meta.get("hqic"),
            "ARDL_RMSE_trainval": split_metrics.get("RMSE_trainval"),
            "ARDL_RMSE_test": split_metrics.get("RMSE_test"),
            "ARDL_R2_trainval": split_metrics.get("R2_trainval"),
            "ARDL_R2_test": split_metrics.get("R2_test"),
            "ARDL_durbin_watson": diag.get("durbin_watson"),
            "ARDL_ljungbox_q10": diag.get("ljungbox_q10"),
            "ARDL_ljungbox_p": diag.get("ljungbox_p_q10"),
            # ── LSTM ──
            "LSTM_lookback": int(sel["Lookback"]),
            "LSTM_batch_size": int(sel["Batch_size"]),
            "LSTM_best_epoch": int(sel["Best_Epoch"]),
            "LSTM_input_channels": (int(k) + 1) if k is not None else None,
            "LSTM_units": str(lstm_cfg.get("lstm_units")),
            "LSTM_dense_units": str(lstm_cfg.get("dense_units")),
            "LSTM_dropout": lstm_cfg.get("dropout_rate"),
            "LSTM_lr": lstm_cfg.get("learning_rate"),
            "LSTM_max_epochs": lstm_cfg.get("epochs"),
            "LSTM_es_patience": lstm_cfg.get("early_stopping_patience"),
            "LSTM_rlr_patience": lstm_cfg.get("reduce_lr_patience"),
            "LSTM_Val_RMSE": float(sel["Val_RMSE"]),
            "LSTM_Train_RMSE": float(sel["Train_RMSE"]),
            "LSTM_Test_RMSE": summary.get("lstm", {}).get("LSTM_RMSE"),
            "LSTM_train_samples": int(sel["Train_samples"]),
            "LSTM_val_samples": int(sel["Val_samples"]),
        })
    return pd.DataFrame(rows)


# ── Statistical comparison builders ──────────────────────────────────────────

def build_baseline_tests(series: dict) -> pd.DataFrame:
    """ARDL(per CEV) vs Persistence and AR(1): DM + paired Wilcoxon."""
    if "Persistence" not in series or "AR1" not in series:
        raise FileNotFoundError(
            "Baseline prediction series missing. Run experiments/run_baselines.py first.")

    rows = []
    for cev, label in zip(CEV_ORDER, PCA_LABELS):
        key = f"ARDL::{label}"
        if key not in series:
            continue
        m = series[key]
        for bname in ["Persistence", "AR1"]:
            b = series[bname]
            idx = m.index.intersection(b.index)
            if len(idx) == 0:
                continue
            actual = m.loc[idx, "Actual_VNINDEX"].to_numpy()
            f_model = m.loc[idx, "Predicted_VNINDEX"].to_numpy()
            f_base = b.loc[idx, "Predicted_VNINDEX"].to_numpy()

            dm_mse = diebold_mariano_test(actual, f_model, f_base, loss_type="mse")
            dm_mae = diebold_mariano_test(actual, f_model, f_base, loss_type="mae")
            wx = wilcoxon_abs_error(actual, f_model, f_base)

            rmse_m = float(np.sqrt(np.mean((actual - f_model) ** 2)))
            rmse_b = float(np.sqrt(np.mean((actual - f_base) ** 2)))
            rows.append({
                "CEV": cev, "comparison": f"ARDL vs {bname}",
                "n": len(idx),
                "RMSE_model": rmse_m, "RMSE_baseline": rmse_b,
                "RMSE_gain": rmse_b - rmse_m,
                "RMSE_gain_pct": 100.0 * (rmse_b - rmse_m) / rmse_b,
                "DM_MSE_stat": dm_mse["dm_stat"], "DM_MSE_p": dm_mse["p_value"],
                "DM_MAE_stat": dm_mae["dm_stat"], "DM_MAE_p": dm_mae["p_value"],
                "Wilcoxon_p": wx["p_value"],
                "median_abs_err_diff": wx["median_abs_err_diff"],
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["DM_MSE_p_holm"] = holm_correction(out["DM_MSE_p"].tolist())
        out["Wilcoxon_p_holm"] = holm_correction(out["Wilcoxon_p"].tolist())
        out["significant_5pct_holm"] = out["DM_MSE_p_holm"] < 0.05
    return out


def build_ardl_coefficient_structure(run_dir: Path) -> pd.DataFrame:
    """How the fitted ARDL splits explanatory weight between the AR lag and PCs.

    This is what actually answers "does the representation matter to ARDL":
    the autoregressive coefficient on VNINDEX.L1 versus the number of
    exogenous PC lags that reach significance.
    """
    rows = []
    for label in ALL_LABELS:
        p = run_dir / label / "outputs" / "ardl_vnindex_report" / "ardl_coefficients.csv"
        if not p.exists():
            continue
        c = pd.read_csv(p)
        c = c.rename(columns={c.columns[0]: "term"})
        ar = c[c["term"] == "VNINDEX.L1"]
        exog = c[~c["term"].isin(["const"]) & ~c["term"].str.startswith("VNINDEX")]
        n_exog = len(exog)
        n_sig = int((exog["pvalue"] < 0.05).sum())
        rows.append({
            "label": label,
            "n_exog_terms": n_exog,
            "AR_coef_VNINDEX_L1": float(ar["coef"].iloc[0]) if not ar.empty else np.nan,
            "AR_pvalue": float(ar["pvalue"].iloc[0]) if not ar.empty else np.nan,
            "n_exog_significant_5pct": n_sig,
            "pct_exog_significant": 100.0 * n_sig / n_exog if n_exog else np.nan,
            # At alpha=0.05 you expect this many false positives by chance:
            "expected_false_positives": 0.05 * n_exog,
            "significant_terms": ", ".join(exog.loc[exog["pvalue"] < 0.05, "term"].head(8)),
        })
    return pd.DataFrame(rows)


def build_error_concentration(series: dict, top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How concentrated the Test loss is, and how close ARDL is to a random walk.

    A handful of shock days can dominate the squared-error sum. When that
    happens the DM statistic loses power (the loss-differential series is
    driven by a few observations with a large HAC variance), so the
    concentration must be reported alongside the p-values rather than
    left implicit.
    """
    conc_rows, rw_rows = [], []
    for label in ALL_LABELS:
        key = f"ARDL::{label}"
        if key not in series:
            continue
        d = series[key]
        y = d["Actual_VNINDEX"].to_numpy()
        f = d["Predicted_VNINDEX"].to_numpy()
        se = (y - f) ** 2
        order = np.argsort(se)[::-1]
        top_idx = order[:top_n]
        keep = np.setdiff1d(np.arange(len(se)), top_idx)

        conc_rows.append({
            "label": label,
            "n": len(se),
            "RMSE_all": float(np.sqrt(se.mean())),
            f"RMSE_excl_top{top_n}": float(np.sqrt(se[keep].mean())),
            f"top{top_n}_share_of_SSE_pct": float(100 * se[top_idx].sum() / se.sum()),
            "worst_day": str(d.index[top_idx[0]].date()),
            "worst_day_actual": float(y[top_idx[0]]),
            "worst_day_pred": float(f[top_idx[0]]),
        })

        # Similarity to the random walk: if ARDL is essentially reproducing
        # Persistence, no amount of PC engineering will change the forecast.
        if "Persistence" in series:
            p = series["Persistence"]
            idx = d.index.intersection(p.index)
            fa = d.loc[idx, "Predicted_VNINDEX"].to_numpy()
            fp = p.loc[idx, "Predicted_VNINDEX"].to_numpy()
            rw_rows.append({
                "label": label,
                "corr_with_persistence": float(np.corrcoef(fa, fp)[0, 1]),
                "mean_abs_diff": float(np.mean(np.abs(fa - fp))),
                "max_abs_diff": float(np.max(np.abs(fa - fp))),
            })
    return pd.DataFrame(conc_rows), pd.DataFrame(rw_rows)


def build_ardl_vs_lstm_tests(series: dict) -> pd.DataFrame:
    """ARDL vs LSTM within each representation (identical test window)."""
    rows = []
    for label in ALL_LABELS:
        ka, kl = f"ARDL::{label}", f"LSTM::{label}"
        if ka not in series or kl not in series:
            continue
        a, l = series[ka], series[kl]
        idx = a.index.intersection(l.index)
        if len(idx) == 0:
            continue
        actual = a.loc[idx, "Actual_VNINDEX"].to_numpy()
        f_ardl = a.loc[idx, "Predicted_VNINDEX"].to_numpy()
        f_lstm = l.loc[idx, "Predicted_VNINDEX"].to_numpy()

        dm_mse = diebold_mariano_test(actual, f_ardl, f_lstm, loss_type="mse")
        dm_mae = diebold_mariano_test(actual, f_ardl, f_lstm, loss_type="mae")
        wx = wilcoxon_abs_error(actual, f_ardl, f_lstm)
        rows.append({
            "label": label, "n": len(idx),
            "ARDL_RMSE": float(np.sqrt(np.mean((actual - f_ardl) ** 2))),
            "LSTM_RMSE": float(np.sqrt(np.mean((actual - f_lstm) ** 2))),
            "DM_MSE_stat": dm_mse["dm_stat"], "DM_MSE_p": dm_mse["p_value"],
            "DM_MAE_stat": dm_mae["dm_stat"], "DM_MAE_p": dm_mae["p_value"],
            "Wilcoxon_p": wx["p_value"],
            "winner": "ARDL" if dm_mse["mean_diff"] < 0 else "LSTM",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["DM_MSE_p_holm"] = holm_correction(out["DM_MSE_p"].tolist())
    return out


def build_pca_vs_nodr_tests(series: dict) -> pd.DataFrame:
    """RQ3: each PCA representation vs the no-reduction baseline, per model."""
    rows = []
    for model in ["ARDL", "LSTM"]:
        base_key = f"{model}::{NO_DR_LABEL}"
        if base_key not in series:
            continue
        b = series[base_key]
        for cev, label in zip(CEV_ORDER, PCA_LABELS):
            key = f"{model}::{label}"
            if key not in series:
                continue
            m = series[key]
            idx = m.index.intersection(b.index)
            if len(idx) == 0:
                continue
            actual = m.loc[idx, "Actual_VNINDEX"].to_numpy()
            f_pca = m.loc[idx, "Predicted_VNINDEX"].to_numpy()
            f_nodr = b.loc[idx, "Predicted_VNINDEX"].to_numpy()
            dm = diebold_mariano_test(actual, f_pca, f_nodr, loss_type="mse")
            wx = wilcoxon_abs_error(actual, f_pca, f_nodr)
            rmse_p = float(np.sqrt(np.mean((actual - f_pca) ** 2)))
            rmse_n = float(np.sqrt(np.mean((actual - f_nodr) ** 2)))
            rows.append({
                "model": model, "CEV": cev, "n": len(idx),
                "RMSE_pca": rmse_p, "RMSE_no_dr": rmse_n,
                "RMSE_gain_pct": 100.0 * (rmse_n - rmse_p) / rmse_n,
                "DM_MSE_stat": dm["dm_stat"], "DM_MSE_p": dm["p_value"],
                "Wilcoxon_p": wx["p_value"],
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["DM_MSE_p_holm"] = holm_correction(out["DM_MSE_p"].tolist())
        out["significant_5pct_holm"] = out["DM_MSE_p_holm"] < 0.05
    return out


# ── Figures ──────────────────────────────────────────────────────────────────

def fig_scree(pca: dict, out_dir: Path) -> None:
    sp = pca["spectrum"]
    dim = pca["dimensionality"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    n = len(sp)
    ax1.bar(range(1, n + 1), sp["explained_variance_pct"], color=PALETTE["ardl"], alpha=.85)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_title(f"(a) Scree plot — PC1..PC{n} (retained at CEV=0.95)")
    ax1.set_xticks(range(1, n + 1))
    for i, v in enumerate(sp["explained_variance_pct"][:6], start=1):
        ax1.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    ax1.grid(axis="y", alpha=.3)

    ax2.plot(range(1, n + 1), sp["cumulative_variance_pct"], "o-",
             color=PALETTE["ardl"], lw=2, ms=5, label="Cumulative variance")
    for _, r in dim.iterrows():
        ax2.axhline(r["CEV_requested"] * 100, ls=":", lw=1, color=PALETTE["warn"], alpha=.7)
        ax2.plot(r["k"], r["CEV_achieved"] * 100, "s", ms=9,
                 color=PALETTE["accent"], zorder=5)
        ax2.annotate(f"CEV {r['CEV_requested']:.2f}\nk={int(r['k'])}",
                     (r["k"], r["CEV_achieved"] * 100),
                     textcoords="offset points", xytext=(6, -14), fontsize=8)
    ax2.set_xlabel("Number of retained components (k)")
    ax2.set_ylabel("Cumulative explained variance (%)")
    ax2.set_title("(b) CEV threshold → retained dimensionality")
    ax2.set_xticks(range(1, n + 1))
    ax2.grid(alpha=.3)
    ax2.legend(loc="lower right", fontsize=9)

    fig.suptitle("Figure 1. PCA spectrum and the CEV→k mapping (train-only fit)", y=1.02)
    _save_fig(fig, out_dir, "fig01_pca_scree_cev")


def fig_loadings(pca: dict, out_dir: Path) -> None:
    L = pca["loadings"]
    if L is None or L.shape[1] < 6:
        print("  [WARN] fewer than 6 retained PCs available; skipping loading figure")
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for i, pc in enumerate(list(L.columns)[:6]):
        ax = axes[i // 3][i % 3]
        vals = L[pc].sort_values()
        ax.bar(range(len(vals)), vals.to_numpy(),
               color=np.where(vals.to_numpy() > 0, PALETTE["ardl"], PALETTE["warn"]))
        frac_pos = float((L[pc] > 0).mean())
        ax.axhline(0, color="k", lw=.8)
        ax.set_title(f"{pc} — {frac_pos*100:.1f}% positive loadings", fontsize=10)
        ax.set_xlabel("Stocks (sorted by loading)")
        ax.set_ylabel("Loading")
        ax.grid(axis="y", alpha=.3)
    fig.suptitle("Figure 2. Loading structure of PC1–PC6 (318 stocks): PC1 is a "
                 "market-wide factor, PC2+ are contrast factors", y=1.01)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig02_pc_loadings")


def fig_performance_vs_k(res: pd.DataFrame, ms: pd.DataFrame, out_dir: Path) -> None:
    pca = res[res["label"].isin(PCA_LABELS)].copy().sort_values("CEV_requested")
    nodr = res[res["label"] == NO_DR_LABEL]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.plot(pca["k"], pca["ARDL_RMSE"], "-", color="#333333", lw=1.4, zorder=1)
    for _, r in pca.iterrows():
        ax.plot(r["k"], r["ARDL_RMSE"], marker=REP_STYLE[r["label"]]["marker"],
                color=_rep_color(r["label"]), ms=11, mec="white", mew=0.8, zorder=3)
    if not nodr.empty:
        ax.axhline(float(nodr["ARDL_RMSE"].iloc[0]), ls="--", color=PALETTE["neutral"],
                   label=f"no-DR (k=318): {float(nodr['ARDL_RMSE'].iloc[0]):.1f}")
    ax.set_xlabel("k (retained components)")
    ax.set_ylabel("Test RMSE (index points)")
    ax.set_title("(a) ARDL — near-invariant across k")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="center right")

    # Inset: the PCA-only range is ~0.09 index points wide and invisible on
    # the shared axis, but it carries the O2/RQ1 answer for ARDL.
    inset = ax.inset_axes([0.12, 0.30, 0.52, 0.42])
    inset.plot(pca["k"], pca["ARDL_RMSE"], "-", color="#333333", lw=1.2, zorder=1)
    for _, r in pca.iterrows():
        inset.plot(r["k"], r["ARDL_RMSE"], marker=REP_STYLE[r["label"]]["marker"],
                   color=_rep_color(r["label"]), ms=8, mec="white", mew=0.6, zorder=3)
        inset.annotate(f"{r['CEV_requested']:.2f}", (r["k"], r["ARDL_RMSE"]),
                       textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    lo, hi = pca["ARDL_RMSE"].min(), pca["ARDL_RMSE"].max()
    pad = (hi - lo) * 0.55 or 0.05
    inset.set_ylim(lo - pad, hi + pad)
    inset.set_title("zoom: PCA only", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=.3)

    ax = axes[1]
    ax.plot(pca["k"], pca["LSTM_RMSE"], "-", color="#333333", lw=1.4, zorder=1,
            label="LSTM (single seed 42)")
    for _, r in pca.iterrows():
        ax.plot(r["k"], r["LSTM_RMSE"], marker=REP_STYLE[r["label"]]["marker"],
                color=_rep_color(r["label"]), ms=11, mec="white", mew=0.8, zorder=3)
        tag = " *collapsed*" if r["label"] in ("pca_cev_0.80", "pca_cev_0.85") else ""
        ax.annotate(f"{r['CEV_requested']:.2f}{tag}", (r["k"], r["LSTM_RMSE"]),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    if not nodr.empty:
        ax.axhline(float(nodr["LSTM_RMSE"].iloc[0]), ls="--", color=PALETTE["neutral"],
                   label=f"no-DR (k=318): {float(nodr['LSTM_RMSE'].iloc[0]):.1f}")
    ax.set_xlabel("k (retained components)")
    ax.set_ylabel("Test RMSE (index points)")
    ax.set_title("(b) LSTM — erratic (training instability)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    order = [l for l in ALL_LABELS if l in set(ms["label"])]
    data = [ms[ms["label"] == l]["RMSE"].to_numpy() for l in order]
    bp = ax.boxplot(data, labels=[l.replace("pca_cev_", "CEV ") for l in order],
                    patch_artist=True, widths=.6)
    for patch, lab in zip(bp["boxes"], order):
        patch.set_facecolor(_rep_color(lab))
        patch.set_alpha(.55)
    for i, (arr, lab) in enumerate(zip(data, order), start=1):
        ax.plot([i] * len(arr), arr, marker=REP_STYLE.get(lab, {}).get("marker", "o"),
                linestyle="none", color="black", ms=5, alpha=.85)
    ax.set_ylabel("Test RMSE across seeds")
    ax.set_title("(c) LSTM multi-seed spread (42/52/62/72/82)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=.3)

    fig.suptitle("Figure 3. Out-of-sample Test RMSE versus representation "
                 "dimensionality (identical Test window, N=167, T+1)", y=1.03)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig03_performance_vs_k")


def fig_baselines(bt: pd.DataFrame, res: pd.DataFrame, out_dir: Path) -> None:
    if bt.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    pers = bt[bt["comparison"] == "ARDL vs Persistence"].sort_values("CEV")
    ar1 = bt[bt["comparison"] == "ARDL vs AR1"].sort_values("CEV")

    x = np.arange(len(pers))
    w = .35
    ax1.bar(x - w / 2, pers["RMSE_gain_pct"], w, label="vs Persistence",
            color=PALETTE["ardl"])
    ax1.bar(x + w / 2, ar1["RMSE_gain_pct"], w, label="vs AR(1)", color=PALETTE["accent"])
    ax1.axhline(0, color="k", lw=.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c:.2f}\n(k={int(k)})" for c, k in
                         zip(pers["CEV"], res.set_index("CEV_requested")
                             .reindex(pers["CEV"])["k"])])
    ax1.set_xlabel("CEV threshold")
    ax1.set_ylabel("RMSE improvement (%)")
    ax1.set_title("(a) PCA-ARDL incremental value over naive baselines")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=.3)
    for xi, v in zip(x - w / 2, pers["RMSE_gain_pct"]):
        ax1.text(xi, v + .03, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, ar1["RMSE_gain_pct"]):
        ax1.text(xi, v + .03, f"{v:.2f}", ha="center", fontsize=8)

    ax2.plot(pers["CEV"], pers["DM_MSE_p"], "o-", color=PALETTE["ardl"],
             label="DM(MSE) vs Persistence")
    ax2.plot(ar1["CEV"], ar1["DM_MSE_p"], "s-", color=PALETTE["accent"],
             label="DM(MSE) vs AR(1)")
    ax2.plot(pers["CEV"], pers["Wilcoxon_p"], "^--", color=PALETTE["lstm"],
             label="Wilcoxon vs Persistence")
    ax2.axhline(.05, ls=":", color=PALETTE["warn"], lw=2, label="alpha = 0.05")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("CEV threshold")
    ax2.set_ylabel("p-value")
    ax2.set_title("(b) No comparison clears alpha=0.05")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=.3)

    fig.suptitle("Figure 4. PCA-ARDL versus Persistence and AR(1) on the same "
                 "Test window (N=167)", y=1.03)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig04_baselines")


def fig_ar_coefficient(coef: pd.DataFrame, res: pd.DataFrame, out_dir: Path) -> None:
    """The AR-lag vs PC trade-off as k grows — the mechanism behind RQ1/RQ3."""
    if coef.empty:
        return
    kmap = res.set_index("label")["k"].to_dict()
    c = coef.copy()
    c["k"] = c["label"].map(kmap).astype(float)
    c = c.sort_values("k")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    ax1.semilogx(c["k"], c["AR_coef_VNINDEX_L1"], "-", color="#333333", lw=1.4, zorder=1)
    for _, r in c.iterrows():
        ax1.plot(r["k"], r["AR_coef_VNINDEX_L1"],
                 marker=REP_STYLE.get(r["label"], {"marker": "P"})["marker"],
                 color=_rep_color(r["label"]), ms=11, mec="white", mew=0.8, zorder=3)
        ax1.annotate(f"k={int(r['k'])}", (r["k"], r["AR_coef_VNINDEX_L1"]),
                     textcoords="offset points", xytext=(4, 8), fontsize=8)
    ax1.axhline(1.0, ls=":", color=PALETTE["neutral"],
                label="Random walk (coef = 1)")
    ax1.set_xlabel("k (log scale)")
    ax1.set_ylabel("Coefficient on VNINDEX.L1")
    ax1.set_title("(a) The autoregressive anchor erodes as k grows")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=.3, which="both")

    pca = c[c["label"] != NO_DR_LABEL]
    x = np.arange(len(pca))
    ax2.bar(x, pca["n_exog_significant_5pct"], .55, color=PALETTE["accent"],
            alpha=.85, label="PC lags significant at 5%")
    ax2.plot(x, pca["n_exog_terms"], "ks--", ms=6, label="PC lags available")
    ax2.plot(x, pca["expected_false_positives"], "v:", color=PALETTE["warn"],
             ms=6, label="Expected by chance (0.05k)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"k={int(k)}" for k in pca["k"]])
    ax2.set_ylabel("Number of exogenous lag terms")
    ax2.set_title("(b) More components → more usable exogenous signal")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=.3)

    fig.suptitle("Figure 8. ARDL coefficient structure: the PCs supplement, never "
                 "replace, the autoregressive term", y=1.02)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig08_ardl_coefficients")


def fig_error_concentration(series: dict, out_dir: Path, label: str = "pca_cev_0.90") -> None:
    """Why the DM tests have no power: the loss is a few shock days."""
    key = f"ARDL::{label}"
    if key not in series or "Persistence" not in series:
        return
    d, p = series[key], series["Persistence"]
    idx = d.index.intersection(p.index)
    y = d.loc[idx, "Actual_VNINDEX"].to_numpy()
    fa = d.loc[idx, "Predicted_VNINDEX"].to_numpy()
    fp = p.loc[idx, "Predicted_VNINDEX"].to_numpy()

    se_a, se_p = (y - fa) ** 2, (y - fp) ** 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    ax1.plot(idx, se_a, color=PALETTE["ardl"], lw=1.2, label="ARDL squared error")
    ax1.set_yscale("symlog", linthresh=1)
    ax1.set_ylabel("Squared error (symlog)")
    ax1.set_xlabel("Date")
    ax1.set_title("(a) Test loss is dominated by the April-2025 shock")
    ax1.tick_params(axis="x", rotation=25)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=.3)

    share = np.sort(se_a)[::-1].cumsum() / se_a.sum() * 100
    ax2.plot(range(1, len(share) + 1), share, color=PALETTE["accent"], lw=2)
    for n in (5, 10, 20):
        if n <= len(share):
            ax2.plot(n, share[n - 1], "o", color=PALETTE["warn"], ms=7)
            ax2.annotate(f"top {n} days: {share[n-1]:.0f}%", (n, share[n - 1]),
                         textcoords="offset points", xytext=(10, -4), fontsize=9)
    ax2.axhline(80, ls=":", color=PALETTE["neutral"])
    ax2.set_xlabel("Number of worst days (sorted)")
    ax2.set_ylabel("Cumulative share of total squared error (%)")
    ax2.set_title("(b) Loss concentration curve")
    ax2.set_xlim(0, min(60, len(share)))
    ax2.grid(alpha=.3)

    corr = np.corrcoef(fa, fp)[0, 1]
    fig.suptitle(f"Figure 9. Error concentration for ARDL at {label} — "
                 f"corr(ARDL, Persistence) = {corr:.5f}", y=1.02)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig09_error_concentration")


def fig_multiseed(ms: pd.DataFrame, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    order = [l for l in ALL_LABELS if l in set(ms["label"])]
    names = [l.replace("pca_cev_", "CEV ").replace("no_dr", "no-DR") for l in order]

    colors = [_rep_color(l) for l in order]
    means = [ms[ms["label"] == l]["RMSE"].mean() for l in order]
    stds = [ms[ms["label"] == l]["RMSE"].std(ddof=1) for l in order]
    ax1.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=.9,
           error_kw={"ecolor": "#333333", "lw": 1.3})
    ax1.set_ylabel("Test RMSE (mean ± SD, n=5 seeds)")
    ax1.set_title("(a) LSTM stability by representation")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(axis="y", alpha=.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m + s + 2, f"{m:.1f}\n±{s:.1f}", ha="center", fontsize=8)

    cvs = [100 * s / m if m else np.nan for m, s in zip(means, stds)]
    ax2.bar(names, cvs, color=colors, alpha=.9)
    ax2.set_ylabel("Coefficient of variation (%)")
    ax2.set_title("(b) Relative seed dispersion")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", alpha=.3)
    for i, v in enumerate(cvs):
        if not np.isnan(v):
            ax2.text(i, v + .4, f"{v:.1f}%", ha="center", fontsize=8)

    fig.suptitle("Figure 5. LSTM multi-seed stability (seeds 42/52/62/72/82, "
                 "frozen hyperparameters)", y=1.03)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig05_multiseed")


def fig_lstm_collapse(grid: pd.DataFrame, curves: dict, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    pivot = (grid[grid["label"].isin(PCA_LABELS)]
             .assign(cfg=lambda d: "LB" + d["Lookback"].astype(str)
                     + "_BS" + d["Batch_size"].astype(str))
             .pivot(index="cfg", columns="label", values="Best_Epoch"))
    pivot = pivot.reindex(columns=[c for c in PCA_LABELS if c in pivot.columns])
    # "viridis" is perceptually uniform and colorblind-safe (unlike RdYlGn,
    # whose red/green channels are indistinguishable for red-green CVD).
    im = ax1.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis",
                    vmin=0, vmax=float(np.nanmax(pivot.to_numpy())))
    ax1.set_xticks(range(pivot.shape[1]))
    ax1.set_xticklabels([c.replace("pca_cev_", "CEV ") for c in pivot.columns], rotation=30)
    ax1.set_yticks(range(pivot.shape[0]))
    ax1.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.to_numpy()[i, j]
            if not np.isnan(v):
                # viridis is dark at low values, so low Best_Epoch (collapsed
                # configs) needs light text; high values need dark text.
                ax1.text(j, i, int(v), ha="center", va="center", fontsize=8,
                         color="white" if v <= 20 else "black",
                         fontweight="bold" if v <= 5 else "normal")
    ax1.set_title("(a) Best_Epoch per LSTM config\n(1 = collapsed to an untrained model)")
    fig.colorbar(im, ax=ax1, label="Best_Epoch")

    for label in ("pca_cev_0.80", "pca_cev_0.85", "pca_cev_0.95"):
        color = _rep_color(label)
        per = curves.get(label, {})
        for i, (cfg, vl) in enumerate(sorted(per.items())):
            ax2.plot(range(1, len(vl) + 1), vl, color=color, alpha=.7, lw=1.3,
                     label=label.replace("pca_cev_", "CEV ") if i == 0 else None)
    ax2.axvline(26, ls=":", color="k", lw=1)
    ax2.text(27, ax2.get_ylim()[1] * .8, "premature stop\n(epoch 26)", fontsize=8)
    ax2.set_yscale("log")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("val_loss (scaled units, log axis)")
    ax2.set_title("(b) Validation curves: collapsed runs never start descending")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=.3)

    fig.suptitle("Figure 6. LSTM training instability — an early-stopping artefact, "
                 "not a property of the representation", y=1.03)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig06_lstm_collapse")


def fig_forecast_overlay(series: dict, out_dir: Path) -> None:
    best_ardl = "ARDL::pca_cev_0.90"
    best_lstm = "LSTM::pca_cev_0.75"
    if best_ardl not in series or best_lstm not in series:
        print("  [WARN] overlay series unavailable; skipping forecast overlay")
        return
    a, l = series[best_ardl], series[best_lstm]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)

    ax1.plot(a.index, a["Actual_VNINDEX"], color="k", lw=1.6, label="Actual VN-Index")
    ax1.plot(a.index, a["Predicted_VNINDEX"], color=PALETTE["ardl"], lw=1.3,
             label="ARDL(1,1), CEV=0.90 (k=6)")
    ax1.plot(l.index, l["Predicted_VNINDEX"], color=PALETTE["lstm"], lw=1.3, alpha=.85,
             label="LSTM, CEV=0.75 (k=2)")
    if "Persistence" in series:
        p = series["Persistence"]
        ax1.plot(p.index, p["Predicted_VNINDEX"], color=PALETTE["accent"],
                 lw=1.2, ls="--", alpha=.9, label="Persistence")
    ax1.set_ylabel("VN-Index")
    ax1.set_title("(a) One-step-ahead Test forecasts (2024-08-27 → 2025-04-29)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=.3)

    ax2.plot(a.index, a["Actual_VNINDEX"] - a["Predicted_VNINDEX"],
             color=PALETTE["ardl"], lw=1, label="ARDL residual")
    ax2.plot(l.index, l["Actual_VNINDEX"] - l["Predicted_VNINDEX"],
             color=PALETTE["lstm"], lw=1, alpha=.8, label="LSTM residual")
    ax2.axhline(0, color="k", lw=.8)
    ax2.set_ylabel("Residual (index points)")
    ax2.set_xlabel("Date")
    ax2.set_title("(b) Residuals — LSTM shows a persistent level bias")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=.3)

    fig.suptitle("Figure 7. Best ARDL and best LSTM against the actual index", y=1.0)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig07_forecast_overlay")


# ── Orchestration ────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate analysis tables + figures for a representation-sweep run.")
    ap.add_argument("--run-dir", required=True,
                    help="Parent Run_* directory, e.g. artifacts/Run_20260823_130205")
    ap.add_argument("--out-dir", default=None,
                    help="Output root (default: notebooks/analysis_output/<run_id>)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    _require(run_dir, "run directory")
    manifest = _read_json(_require(run_dir / "sweep_manifest.json", "sweep_manifest.json"))

    out_dir = (Path(args.out_dir) if args.out_dir
               else PROJECT_ROOT / "notebooks" / "analysis_output" / manifest["run_id"])
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _hdr(f"ANALYSIS  |  {manifest['run_id']}")
    print(f"  Run dir : {run_dir}")
    print(f"  Out dir : {out_dir}")
    print(f"  Status  : {manifest['status']}  "
          f"({len(manifest['completed_labels'])}/{len(manifest['planned_labels'])} children)")
    if manifest.get("failed_labels"):
        print(f"  [WARN] failed labels: {manifest['failed_labels']}")

    consolidated: dict = {"run_id": manifest["run_id"], "manifest": manifest}

    # ── Section 0: preprocessing ────────────────────────────────────────────
    _hdr("Section 0 — Preprocessing")
    pre = load_preprocessing(run_dir)
    print(f"  Observations      : {pre['n_obs']}  ({pre['target_start']} → {pre['target_end']})")
    print(f"  Stocks kept/removed: {pre['n_valid_stocks']} / {pre['n_removed_stocks']}")
    print(f"  Scaled feature cols: {pre['n_scaled_cols']}")
    overview = pd.DataFrame([
        {"Item": "Observations (trading days)", "Value": pre["n_obs"]},
        {"Item": "Date range", "Value": f"{pre['target_start']} → {pre['target_end']}"},
        {"Item": "Candidate stocks retained", "Value": pre["n_valid_stocks"]},
        {"Item": "Stocks removed (missing > 20%)", "Value": pre["n_removed_stocks"]},
        {"Item": "Feature columns after scaling", "Value": pre["n_scaled_cols"]},
        {"Item": "Symbols with IQR outliers", "Value": pre["n_outlier_symbols"]},
        {"Item": "Total outlier observations", "Value": pre["total_outliers"]},
        {"Item": "VN-Index mean", "Value": round(pre["target_mean"], 2)},
        {"Item": "VN-Index SD", "Value": round(pre["target_std"], 2)},
        {"Item": "VN-Index min / max",
         "Value": f"{pre['target_min']:.2f} / {pre['target_max']:.2f}"},
    ])
    _save_table(overview, out_dir, "t00_preprocessing_overview",
                caption="Table 0a. Preprocessing overview")
    _save_table(pre["splits"], out_dir, "t00_splits",
                caption="Table 0b. Chronological, non-overlapping splits")
    _save_table(pre["corr_summary"], out_dir, "t00_correlation",
                caption="Table 0c. Cross-sectional correlation of the 318 retained stocks")
    _save_table(pre["missing_dist"], out_dir, "t00_missing",
                caption="Table 0d. Missing-value distribution before filtering")
    consolidated["preprocessing"] = {k: v for k, v in pre.items()
                                     if not isinstance(v, pd.DataFrame)}

    # ── Section 1: PCA structure ────────────────────────────────────────────
    _hdr("Section 1 — PCA representation structure (O1 / RQ1)")
    pca = load_pca_structure(run_dir)
    print(pca["dimensionality"].to_string(index=False))
    _save_table(pca["dimensionality"], out_dir, "t01_dimensionality",
                caption="Table 1a. CEV threshold → retained dimensionality (train-only fit)")
    _save_table(pca["spectrum"], out_dir, "t01_spectrum",
                caption="Table 1b. Eigenvalue spectrum of the retained components")
    _save_table(pca["threshold_table"], out_dir, "t01_threshold_map",
                caption="Table 1c. Full CEV→k map produced by the PCA stage")

    if pca["loadings"] is not None:
        L = pca["loadings"]
        rows = []
        for pc in list(L.columns)[:6]:
            s = L[pc]
            rows.append({
                "PC": pc,
                "explained_var_pct": float(
                    pca["spectrum"].set_index("PC").loc[pc, "explained_variance_pct"]),
                "cumulative_pct": float(
                    pca["spectrum"].set_index("PC").loc[pc, "cumulative_variance_pct"]),
                "mean_loading": float(s.mean()),
                "sd_loading": float(s.std()),
                "pct_positive": float((s > 0).mean() * 100),
                "top_positive": ", ".join(s.nlargest(3).index),
                "top_negative": ", ".join(s.nsmallest(3).index),
                "interpretation": ("Market-wide common factor" if (s > 0).mean() > .85
                                   else "Contrast / rotation factor"),
            })
        pcstruct = pd.DataFrame(rows)
        _save_table(pcstruct, out_dir, "t01_pc_interpretation",
                    caption="Table 1d. Structure and interpretation of PC1–PC6")
        consolidated["pc_structure"] = pcstruct.to_dict("records")

    consolidated["dimensionality"] = pca["dimensionality"].to_dict("records")
    fig_scree(pca, out_dir)
    fig_loadings(pca, out_dir)

    # ── Section 2: forecasting performance ──────────────────────────────────
    _hdr("Section 2 — Forecast performance versus dimensionality (O2 / RQ1)")
    res = load_forecast_results(run_dir)
    perf = res[["label", "k", "CEV_requested", "ARDL_P", "ARDL_Q", "ARDL_RMSE",
                "ARDL_MAE", "ARDL_R2", "LSTM_LB", "LSTM_BS", "LSTM_best_epoch",
                "LSTM_RMSE", "LSTM_MAE", "LSTM_R2", "N_test"]]
    print(perf.to_string(index=False))
    _save_table(perf, out_dir, "t02_performance",
                caption="Table 2. Test-set performance of the selected ARDL and LSTM models")
    consolidated["performance"] = perf.to_dict("records")

    ms = load_multiseed(run_dir)
    _save_table(ms, out_dir, "t04_multiseed_per_seed",
                caption="Table 4a. Per-seed LSTM Test metrics")
    ms_summary = (ms.groupby("label")
                  .agg(RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", lambda s: s.std(ddof=1)),
                       MAE_mean=("MAE", "mean"), MAE_sd=("MAE", lambda s: s.std(ddof=1)),
                       R2_mean=("R2", "mean"), R2_sd=("R2", lambda s: s.std(ddof=1)),
                       n_seeds=("Seed", "count"))
                  .reindex([l for l in ALL_LABELS if l in set(ms["label"])])
                  .reset_index())
    ms_summary["RMSE_CV_pct"] = 100 * ms_summary["RMSE_sd"] / ms_summary["RMSE_mean"]
    _save_table(ms_summary, out_dir, "t04_multiseed_summary",
                caption="Table 4b. LSTM multi-seed stability (mean ± SD, n=5)")
    consolidated["multiseed"] = ms_summary.to_dict("records")

    fig_performance_vs_k(res, ms, out_dir)

    # ── LSTM training diagnostics (supports Sections 2 and 4) ───────────────
    _hdr("Section 2b — LSTM training stability audit")
    grid, curves = load_lstm_training_diagnostics(run_dir)
    collapse = (grid.groupby("label")
                .agg(n_configs=("collapsed", "size"),
                     n_collapsed=("collapsed", "sum"),
                     min_val_rmse=("Val_RMSE", "min"),
                     max_val_rmse=("Val_RMSE", "max"))
                .reindex([l for l in ALL_LABELS if l in set(grid["label"])])
                .reset_index())
    collapse["collapse_rate_pct"] = 100 * collapse["n_collapsed"] / collapse["n_configs"]
    print(collapse.to_string(index=False))
    _save_table(collapse, out_dir, "t02b_lstm_collapse",
                caption="Table 2b. LSTM premature-stop incidence "
                        f"(Best_Epoch <= {COLLAPSE_EPOCH_THRESHOLD})")
    _save_table(grid.drop(columns=[c for c in grid.columns if c.startswith("Train_period")
                                   or c.startswith("Val_period")]),
                out_dir, "t02b_lstm_grid",
                caption="Table 2c. Full LSTM tuning grid across representations")
    consolidated["lstm_collapse"] = collapse.to_dict("records")
    fig_lstm_collapse(grid, curves, out_dir)

    # ── Section 3: baselines ────────────────────────────────────────────────
    _hdr("Section 3 — ARDL versus AR(1) and Persistence (RQ3)")
    series = load_prediction_series(run_dir)
    bt = build_baseline_tests(series)
    print(bt[["CEV", "comparison", "RMSE_model", "RMSE_baseline", "RMSE_gain_pct",
              "DM_MSE_p", "Wilcoxon_p", "DM_MSE_p_holm"]].to_string(index=False))
    _save_table(bt, out_dir, "t03_baseline_tests",
                caption="Table 3. PCA-ARDL versus naive baselines: DM and paired Wilcoxon")
    consolidated["baseline_tests"] = bt.to_dict("records")
    fig_baselines(bt, res, out_dir)

    coef = build_ardl_coefficient_structure(run_dir)
    print()
    print(coef.drop(columns=["significant_terms"]).to_string(index=False))
    _save_table(coef, out_dir, "t03b_ardl_coefficients",
                caption="Table 3b. Where the fitted ARDL puts its explanatory weight")
    consolidated["ardl_coefficients"] = coef.to_dict("records")
    fig_ar_coefficient(coef, res, out_dir)

    conc, rw = build_error_concentration(series)
    print()
    print(conc.to_string(index=False))
    _save_table(conc, out_dir, "t03c_error_concentration",
                caption="Table 3c. Concentration of Test squared error in shock days")
    if not rw.empty:
        print()
        print(rw.to_string(index=False))
        _save_table(rw, out_dir, "t03d_random_walk_similarity",
                    caption="Table 3d. How close each PCA-ARDL forecast is to the random walk")
    consolidated["error_concentration"] = conc.to_dict("records")
    consolidated["random_walk_similarity"] = rw.to_dict("records")
    fig_error_concentration(series, out_dir)

    # ── Section 5: DM tests ─────────────────────────────────────────────────
    _hdr("Section 5 — Diebold-Mariano comparisons (O3 / RQ2)")
    al = build_ardl_vs_lstm_tests(series)
    print(al.to_string(index=False))
    _save_table(al, out_dir, "t05_ardl_vs_lstm",
                caption="Table 5a. ARDL versus LSTM within each representation")

    pv = build_pca_vs_nodr_tests(series)
    if not pv.empty:
        print(pv.to_string(index=False))
        _save_table(pv, out_dir, "t05_pca_vs_nodr",
                    caption="Table 5b. Each PCA representation versus the no-reduction baseline")
    consolidated["ardl_vs_lstm"] = al.to_dict("records")
    consolidated["pca_vs_nodr"] = pv.to_dict("records")

    ms_dm = []
    for label in ALL_LABELS:
        p = run_dir / label / "outputs" / "lstm_vnindex_multiseed" / "multiseed_dm_diagnostic.csv"
        if p.exists():
            d = pd.read_csv(p)
            d.insert(0, "label", label)
            ms_dm.append(d)
    if ms_dm:
        ms_dm_df = pd.concat(ms_dm, ignore_index=True)
        _save_table(ms_dm_df, out_dir, "t05_multiseed_dm",
                    caption="Table 5c. Per-seed DM diagnostics (ARDL vs LSTM), reported independently")

    fig_multiseed(ms, out_dir)
    fig_forecast_overlay(series, out_dir)

    # ── Section 6: architecture record ──────────────────────────────────────
    _hdr("Section 6 — Training architecture record (CEV 0.75 and 0.85)")
    arch = load_architecture_record(run_dir, ["pca_cev_0.75", "pca_cev_0.85"])
    print(arch.T.to_string())
    _save_table(arch.T.reset_index().rename(columns={"index": "Parameter", 0: "CEV_0.75",
                                                     1: "CEV_0.85"}),
                out_dir, "t06_architecture",
                caption="Table 6. Exact trained configuration at CEV=0.75 and CEV=0.85")
    consolidated["architecture"] = arch.to_dict("records")

    # ── Consolidated numbers ────────────────────────────────────────────────
    with open(out_dir / "analysis_data.json", "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, default=str)
    print(f"\n  [SAVED] {out_dir / 'analysis_data.json'}")

    _hdr("ANALYSIS COMPLETE")
    print(f"  Tables  : {out_dir / 'tables'}")
    print(f"  Figures : {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
