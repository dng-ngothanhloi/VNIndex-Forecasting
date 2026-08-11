from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


# --- from step_03_load_data.py ---
def run_load_data(context: dict) -> dict:
    """Load PCA features and VNINDEX target."""
    PCA_DIR = context["PCA_DIR"]
    CORE_DIR = context["CORE_DIR"]
    PROJECT_ROOT = context["PROJECT_ROOT"]

    # Load PCA features and VNINDEX target
    train_pca = pd.read_csv(PCA_DIR / "train_pca.csv", parse_dates=["Ngày"]).set_index("Ngày")
    val_pca = pd.read_csv(PCA_DIR / "val_pca.csv", parse_dates=["Ngày"]).set_index("Ngày")
    test_pca = pd.read_csv(PCA_DIR / "test_pca.csv", parse_dates=["Ngày"]).set_index("Ngày")
    vnindex = pd.read_csv(CORE_DIR / "vnindex_target.csv", parse_dates=["Ngày"]).set_index("Ngày")

    train_df = train_pca.join(vnindex, how="inner")
    val_df = val_pca.join(vnindex, how="inner")
    test_df = test_pca.join(vnindex, how="inner")

    pc_cols = [c for c in train_df.columns if c != "VNINDEX"]
    target_col = "VNINDEX"

    # Determine representation method for reporting
    _cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    _reduction_method = "pca"
    _use_overlap = False
    if _cfg_path.exists():
        with open(_cfg_path, "r", encoding="utf-8") as _f:
            _full_cfg = yaml.safe_load(_f)
            _use_overlap = _full_cfg.get("preprocess", {}).get("use_overlap_val", False)
            _reduction_method = _full_cfg.get("reduction", {}).get("method", "pca")

    _repr_label = "PCs" if _reduction_method == "pca" else "features"

    print("Shapes:")
    print("  train_df:", train_df.shape)
    print("  val_df  :", val_df.shape)
    print("  test_df :", test_df.shape)
    print(f"  number of {_repr_label}:", len(pc_cols))

    # Validate: at least 1 feature column
    assert len(pc_cols) >= 1, f"No feature columns found. Expected at least 1, got {len(pc_cols)}."

    # Validate monotonic dates
    assert train_df.index.is_monotonic_increasing, "train_df index not sorted"
    assert val_df.index.is_monotonic_increasing,   "val_df index not sorted"
    assert test_df.index.is_monotonic_increasing,  "test_df index not sorted"

    if _use_overlap:
        # Overlap strategy: val is a subset of train dates → intersection is non-empty (expected)
        overlap_count = len(train_df.index.intersection(val_df.index))
        print(f"  [OVERLAP] val overlaps {overlap_count} dates with train (use_overlap_val=true)")
        assert overlap_count > 0, "use_overlap_val=true but no overlap found between train and val"
        assert val_df.index.intersection(test_df.index).empty, "val and test must not overlap"
    else:
        # Non-overlap strategy: all splits must be disjoint
        assert train_df.index.intersection(val_df.index).empty,  "train and val overlap (use_overlap_val=false)"
        assert val_df.index.intersection(test_df.index).empty,   "val and test overlap"
        assert train_df.index.intersection(test_df.index).empty, "train and test overlap"

    print("Date ranges:")
    print("  train:", train_df.index.min().date(), "->", train_df.index.max().date())
    print("  val  :", val_df.index.min().date(), "->", val_df.index.max().date())
    print("  test :", test_df.index.min().date(), "->", test_df.index.max().date())

    context.update({
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "pc_cols": pc_cols,
        "target_col": target_col,
    })
    return context


# --- from step_04_prepare_data.py ---
def make_windowed_data(df: pd.DataFrame, feature_cols: list, target_col: str, lookback: int):
    """Create windowed time-series dataset.

    Every target at position `end_idx` uses feature history strictly at
    positions [end_idx-lookback, end_idx) — i.e. information dated <=
    t-1 predicts y(t). This function itself never violates that boundary;
    P0-4's cross-boundary fix (see `make_cross_boundary_windowed_data`) is
    about WHERE that history is allowed to come from (a preceding split),
    not about changing this per-window boundary.
    """
    x_values = df[feature_cols].astype(float).values
    y_values = df[[target_col]].astype(float).values
    x_windows, y_windows, end_dates = [], [], []
    for end_idx in range(lookback, len(df)):
        x_windows.append(x_values[end_idx - lookback:end_idx])
        y_windows.append(y_values[end_idx])
        end_dates.append(df.index[end_idx])
    return np.array(x_windows), np.array(y_windows), pd.Index(end_dates)


def add_target_history(df: pd.DataFrame, target_col: str, lookback: int):
    """Add target variable history as additional features."""
    target_hist = []
    values = df[target_col].values
    for end_idx in range(lookback, len(df)):
        target_hist.append(values[end_idx - lookback:end_idx].reshape(lookback, 1))
    return np.array(target_hist)


# ─────────────────────────────────────────────────────────────────
# P0-4: CROSS-BOUNDARY HISTORICAL CONTEXT WINDOWING
# ─────────────────────────────────────────────────────────────────
# Scientific rationale (forecasting-protocol-audit.md, P0-4):
# make_windowed_data(df, ..., lookback) independently applied to Val/Test
# discards the first `lookback` rows of each split as targets, because a
# window needs `lookback` rows of history before its first target. This
# silently shrinks Val/Test target counts as lookback grows and makes
# Val/Test target dates inconsistent ACROSS different lookbacks (LB20 and
# LB60 would evaluate on different, non-comparable date ranges).
#
# Fix: build history from the TAIL of the PRECEDING split (never from a
# later split, never leaking beyond t-1), concatenate it in front of the
# current split, and window over the concatenation. Because the
# concatenation always has exactly `lookback` context rows prepended, the
# resulting windows cover EVERY row of the current split as a target,
# regardless of lookback -- giving identical Val/Test target-date sets
# across all lookbacks (D3/D4 requirement), while feature+target history
# for each window remains strictly <= t-1 (no leakage of same-split
# targets across the boundary, since context rows are drawn only from the
# split that already ends before the current split begins).
def make_cross_boundary_windowed_data(
    context_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    lookback: int,
):
    """Window `target_df` using history context = tail(context_df, lookback) + target_df.

    Every returned target date is a date in `target_df` (never in
    `context_df`), so len(result) == len(target_df) for ANY lookback <=
    len(context_df). History for a target at date t is drawn from actual
    observed rows at t-1, t-2, ..., t-lookback, which may span the
    context_df/target_df boundary but never crosses into information
    dated >= t.

    Parameters
    ----------
    context_df : pd.DataFrame
        The preceding split (e.g. Train for Val windowing, or Train+Val
        for Test windowing). Must have >= lookback rows and must end
        strictly before target_df begins (chronologically contiguous).
    target_df : pd.DataFrame
        The split whose rows become forecast targets (e.g. Val or Test).
    feature_cols, target_col, lookback : see `make_windowed_data`.

    Returns
    -------
    (X, y, target_hist, end_dates) : windowed features, targets, target
        history array (from `add_target_history`), and the end_dates
        Index — guaranteed to equal `target_df.index` exactly.
    """
    if len(context_df) < lookback:
        raise ValueError(
            f"context_df has only {len(context_df)} rows, fewer than "
            f"lookback={lookback}; cannot build cross-boundary context."
        )
    context_tail = context_df.tail(lookback)
    if not context_tail.index.max() < target_df.index.min():
        raise ValueError(
            "context_df must end strictly before target_df begins "
            "(chronological, non-overlapping boundary required)."
        )
    combined = pd.concat([context_tail, target_df], axis=0)

    X, y, end_dates = make_windowed_data(combined, feature_cols, target_col, lookback)
    target_hist = add_target_history(combined, target_col, lookback)

    if not end_dates.equals(target_df.index):
        raise AssertionError(
            "Cross-boundary windowing produced end_dates that do not "
            "exactly match target_df.index — this indicates a boundary "
            "bug and must not be silently accepted."
        )

    return X, y, target_hist, end_dates


def run_prepare_data(context: dict) -> dict:
    """Prepare data - scaling and windowing."""
    train_df = context["train_df"]
    val_df = context["val_df"]
    test_df = context["test_df"]
    pc_cols = context["pc_cols"]
    target_col = context["target_col"]

    # Load hyperparameters from config
    # src/forecasting/lstm/data.py -> parents[3] is the project root.
    PROJECT_ROOT = context.get("PROJECT_ROOT", Path(__file__).resolve().parents[3])
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    lstm_cfg = config.get("lstm", {})
    lookback_values = lstm_cfg.get("lookback_values", [30, 45, 60])
    batch_size_values = lstm_cfg.get("batch_size_values", [16, 32, 60])
    epochs = lstm_cfg.get("epochs", 100)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train_raw = train_df[pc_cols].astype(float)
    y_train_raw = train_df[[target_col]].astype(float)
    x_val_raw = val_df[pc_cols].astype(float)
    y_val_raw = val_df[[target_col]].astype(float)
    x_test_raw = test_df[pc_cols].astype(float)
    y_test_raw = test_df[[target_col]].astype(float)

    x_scaler.fit(x_train_raw)
    y_scaler.fit(y_train_raw)

    train_scaled_df = pd.DataFrame(x_scaler.transform(x_train_raw), index=train_df.index, columns=pc_cols)
    val_scaled_df = pd.DataFrame(x_scaler.transform(x_val_raw), index=val_df.index, columns=pc_cols)
    test_scaled_df = pd.DataFrame(x_scaler.transform(x_test_raw), index=test_df.index, columns=pc_cols)

    train_scaled_df[target_col] = y_scaler.transform(y_train_raw).ravel()
    val_scaled_df[target_col] = y_scaler.transform(y_val_raw).ravel()
    test_scaled_df[target_col] = y_scaler.transform(y_test_raw).ravel()

    print("Prepared scaled datasets for sweep experiments.")
    print("Lookback values:", lookback_values)
    print("Batch sizes:", batch_size_values)
    print("Epochs:", epochs)

    context.update({
        "lookback_values": lookback_values,
        "batch_size_values": batch_size_values,
        "epochs": epochs,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "train_scaled_df": train_scaled_df,
        "val_scaled_df": val_scaled_df,
        "test_scaled_df": test_scaled_df,
    })
    return context
