"""
compare_cev_levels.py – Compare ARDL + LSTM results across CEV thresholds
=========================================================================
Run after:  python run_pipeline.py --multi-cev

Reads from:   outputs/cev_{0.85,0.90,0.95}/
Writes to:    outputs/cev_comparison/
              ├── cev_comparison_summary.csv
              └── cev_comparison_plots.png

Usage:
    python compare/compare_cev_levels.py
    python compare/compare_cev_levels.py --thresholds 0.85 0.90 0.95
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# src/evaluation/compare_cev_levels.py -> parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_pca_metrics(cev_dir: Path) -> dict:
    """Return n_components from pca_metrics.csv (index=metric name, col=value)."""
    f = cev_dir / "pca" / "pca_metrics.csv"
    if not f.exists():
        return {"n_components": None}
    df = pd.read_csv(f, index_col=0)
    # file has rows like: k_optimal, cev_threshold, ...
    try:
        return {"n_components": int(df.loc["k_optimal", "value"])}
    except Exception:
        # fallback: first numeric column
        try:
            return {"n_components": int(df.iloc[0, 0])}
        except Exception:
            return {"n_components": None}


def _load_ardl_best(cev_dir: Path) -> dict:
    """Return best ARDL row (by BIC) from sweep_results.csv."""
    f = cev_dir / "ardl_vnindex_pca_sweep" / "sweep_results.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    ok = df[df["Status"] == "OK"]
    if ok.empty:
        return {}
    row = ok.loc[ok["BIC"].idxmin()]
    return {
        "ardl_p":        int(row["P"]),
        "ardl_q":        int(row["Q"]),
        "ardl_bic":      float(row["BIC"]),
        "ardl_rmse_val": float(row["RMSE_val"]) if "RMSE_val" in row else None,
        "ardl_rmse_test":float(row["RMSE_test"]),
        "ardl_mae_test": float(row["MAE_test"]),
    }


def _load_lstm_best(cev_dir: Path) -> dict:
    """Return best LSTM row (by Val_RMSE) from sweep_summary.csv."""
    f = cev_dir / "lstm_vnindex_sweep" / "sweep_summary.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    if df.empty:
        return {}
    row = df.loc[df["Val_RMSE"].idxmin()]
    return {
        "lstm_lookback":  int(row["Lookback"]),
        "lstm_batch":     int(row["Batch_size"]),
        "lstm_best_epoch":row.get("Best_Epoch", None),
        "lstm_rmse_val":  float(row["Val_RMSE"]),
        "lstm_rmse_test": float(row["Test_RMSE"]),
        "lstm_mae_test":  float(row["Test_MAE"]),
    }


def _load_dm(cev_dir: Path) -> dict:
    """Return DM test MSE result."""
    f = cev_dir / "model_comparison" / "dm_test_results.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    mse_row = df[df["Loss_Type"] == "MSE"]
    if mse_row.empty:
        return {}
    r = mse_row.iloc[0]
    return {
        "dm_stat":       float(r["DM_Stat"]),
        "dm_pvalue":     float(r["p_value"]),
        "dm_significant":bool(r["Significant"]),
    }


def load_cev_result(cev: float) -> dict | None:
    """Load all metrics for one CEV level. Returns None if no data found."""
    cev_dir = PROJECT_ROOT / "outputs" / f"cev_{cev:.2f}"
    if not cev_dir.exists():
        return None

    result = {"cev": cev}
    result.update(_load_pca_metrics(cev_dir))
    result.update(_load_ardl_best(cev_dir))
    result.update(_load_lstm_best(cev_dir))
    result.update(_load_dm(cev_dir))
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Model Performance vs PCA Threshold (CEV)", fontsize=14, fontweight="bold")

    x = df["cev"].astype(float)
    xticks = x.tolist()

    # 1 — RMSE comparison
    ax = axes[0, 0]
    if "ardl_rmse_test" in df.columns:
        ax.plot(x, df["ardl_rmse_test"], marker="o", linewidth=2, label="ARDL (test)")
    if "ardl_rmse_val" in df.columns and df["ardl_rmse_val"].notna().any():
        ax.plot(x, df["ardl_rmse_val"], marker="^", linewidth=2, linestyle="--", label="ARDL (val OOS)")
    if "lstm_rmse_test" in df.columns:
        ax.plot(x, df["lstm_rmse_test"], marker="s", linewidth=2, label="LSTM (test)")
    ax.set_xticks(xticks)
    ax.set_xlabel("CEV Threshold")
    ax.set_ylabel("RMSE")
    ax.set_title("Test RMSE by CEV")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2 — Number of PCs
    ax = axes[0, 1]
    if "n_components" in df.columns and df["n_components"].notna().any():
        bars = ax.bar(x, df["n_components"].astype(float), alpha=0.7, color="steelblue", width=0.02)
        for bar, val in zip(bars, df["n_components"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(int(val)) if pd.notna(val) else "?",
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xticks)
    ax.set_xlabel("CEV Threshold")
    ax.set_ylabel("Number of PCs")
    ax.set_title("PCA Dimensionality")
    ax.grid(True, alpha=0.3, axis="y")

    # 3 — DM Test p-value
    ax = axes[1, 0]
    if "dm_pvalue" in df.columns and df["dm_pvalue"].notna().any():
        colors = ["green" if (sig and pd.notna(sig)) else "red"
                  for sig in df.get("dm_significant", [False] * len(df))]
        ax.bar(x, df["dm_pvalue"].astype(float), alpha=0.75, color=colors, width=0.02)
        ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="α = 0.05")
        ax.axhline(0.01, color="orange", linestyle=":", linewidth=1, label="α = 0.01")
        ax.legend(fontsize=8)
    ax.set_xticks(xticks)
    ax.set_xlabel("CEV Threshold")
    ax.set_ylabel("p-value  (DM test, MSE)")
    ax.set_title("Statistical Significance (lower = better)\nGreen = significant")
    ax.grid(True, alpha=0.3, axis="y")

    # 4 — ARDL model complexity (P+Q)
    ax = axes[1, 1]
    if "ardl_p" in df.columns and "ardl_q" in df.columns:
        complexity = df["ardl_p"].astype(float) + df["ardl_q"].astype(float)
        bars = ax.bar(x, complexity, alpha=0.7, color="mediumpurple", width=0.02)
        for bar, row in zip(bars, df.itertuples()):
            p = int(getattr(row, "ardl_p", 0) or 0)
            q = int(getattr(row, "ardl_q", 0) or 0)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"({p},{q})", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xticks)
    ax.set_xlabel("CEV Threshold")
    ax.set_ylabel("ARDL(P + Q)")
    ax.set_title("ARDL Model Complexity")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = out_dir / "cev_comparison_plots.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path


# ── Recommendation ────────────────────────────────────────────────────────────

def recommend(df: pd.DataFrame) -> dict:
    """Select best CEV by lowest ARDL test RMSE; break ties by n_components."""
    if "ardl_rmse_test" not in df.columns or df["ardl_rmse_test"].isna().all():
        return {}
    best_idx = df["ardl_rmse_test"].idxmin()
    row = df.loc[best_idx]
    return {
        "best_cev":         float(row["cev"]),
        "n_pcs":            int(row["n_components"]) if pd.notna(row.get("n_components")) else None,
        "ardl_rmse_test":   float(row["ardl_rmse_test"]),
        "ardl_model":       f"ARDL({int(row.get('ardl_p',0))},{int(row.get('ardl_q',0))})"
                             if pd.notna(row.get("ardl_p")) else "N/A",
        "lstm_rmse_test":   float(row["lstm_rmse_test"]) if pd.notna(row.get("lstm_rmse_test")) else None,
        "dm_significant":   bool(row["dm_significant"]) if pd.notna(row.get("dm_significant")) else None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare results across CEV thresholds")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.85, 0.90, 0.95])
    args = parser.parse_args()

    print("=" * 72)
    print("  CEV LEVEL COMPARISON")
    print("=" * 72)

    rows = []
    for cev in args.thresholds:
        result = load_cev_result(cev)
        if result is None:
            print(f"\n  [NOT FOUND]  CEV={cev:.2f}: no results found — run pipeline first")
            continue
        rows.append(result)
        print(f"\n  CEV = {cev:.2f}")
        n = result.get("n_components")
        print(f"    PCs         : {n if n else '?'}")
        if "ardl_p" in result:
            print(f"    ARDL({result['ardl_p']},{result['ardl_q']})"
                  f"  BIC={result.get('ardl_bic', float('nan')):.2f}"
                  f"  RMSE_val={result.get('ardl_rmse_val', float('nan')):.4f}"
                  f"  RMSE_test={result.get('ardl_rmse_test', float('nan')):.4f}")
        if "lstm_lookback" in result:
            print(f"    LSTM(lb={result['lstm_lookback']},bs={result['lstm_batch']})"
                  f"  Val_RMSE={result.get('lstm_rmse_val', float('nan')):.4f}"
                  f"  Test_RMSE={result.get('lstm_rmse_test', float('nan')):.4f}")
        if "dm_pvalue" in result:
            sig = "[SIGNIFICANT]" if result.get("dm_significant") else "[NOT SIGNIFICANT]"
            print(f"    DM Test     : stat={result['dm_stat']:+.4f}  p={result['dm_pvalue']:.4f}  {sig}")

    if not rows:
        print("\n[FAIL] No CEV results found. Run: python run_pipeline.py --multi-cev")
        sys.exit(1)

    df = pd.DataFrame(rows)

    out_dir = PROJECT_ROOT / "outputs" / "cev_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "cev_comparison_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[COMP] Summary saved: {csv_path}")

    # Plot
    fig_path = plot_comparison(df, out_dir)
    print(f"[COMP] Plots saved:   {fig_path}")

    # Recommendation
    rec = recommend(df)
    if rec:
        print("\n" + "=" * 72)
        print("  RECOMMENDATION")
        print("=" * 72)
        print(f"\n  Best CEV threshold : {rec['best_cev']:.2f}")
        print(f"  PCs selected       : {rec['n_pcs']}")
        print(f"  ARDL model         : {rec['ardl_model']}")
        print(f"  ARDL RMSE_test     : {rec['ardl_rmse_test']:.4f}")
        if rec.get("lstm_rmse_test"):
            print(f"  LSTM RMSE_test     : {rec['lstm_rmse_test']:.4f}")
        if rec.get("dm_significant") is not None:
            sig_str = "Yes — ARDL significantly better" if rec["dm_significant"] else "No"
            print(f"  DM significant     : {sig_str}")
        print()

    print("=" * 72)


if __name__ == "__main__":
    main()
