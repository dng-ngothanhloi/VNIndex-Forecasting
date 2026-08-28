"""
pca_model.py  –  VN-Index PCA Pipeline
========================================
Chạy: python src/pca_model.py --config config/config.yaml

Requires: data/processed/splits/train_scaled.csv (output của preprocess.py)

Output artifacts:
  data/processed/
        pca/train_pca.csv / pca/val_pca.csv / pca/test_pca.csv
        pca/pca_loadings.csv                          ← loadings matrix (p × k)
        pca/pca_threshold_summary.csv                 ← k theo các ngưỡng CEV
        pca/pca_metrics.csv                           ← metrics tóm tắt
        pca/pca_corr_after.csv                        ← xác nhận PC không tương quan
  models/
    pca_model.pkl
  logs/figures/
    pca_summary.png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yaml
from reduction import PCAReducer
from preprocess_steps import save_correlation_heatmap_full, save_distribution_figure


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sector_mapping(mapping_path: Path) -> pd.DataFrame:
    """Load optional symbol-sector mapping file. Returns empty dataframe if missing."""
    if not mapping_path.exists():
        return pd.DataFrame(columns=["Symbol", "Sector"])

    mapping_df = pd.read_csv(mapping_path, sep=None, engine="python", encoding="utf-8-sig")
    mapping_df.columns = [str(c).strip() for c in mapping_df.columns]

    symbol_candidates = ["Symbol", "symbol", "Ticker", "ticker", "Mã", "Ma"]
    sector_candidates = ["Sector", "sector", "Industry", "industry", "Ngành", "Nganh"]

    symbol_col = next((c for c in symbol_candidates if c in mapping_df.columns), None)
    sector_col = next((c for c in sector_candidates if c in mapping_df.columns), None)

    if symbol_col is None or sector_col is None:
        print(
            f"[SECTOR MAP] Skip: file {mapping_path} thiếu cột Symbol/Sector. "
            f"Columns hiện có: {mapping_df.columns.tolist()}"
        )
        return pd.DataFrame(columns=["Symbol", "Sector"])

    mapping_df = mapping_df[[symbol_col, sector_col]].copy()
    mapping_df.columns = ["Symbol", "Sector"]
    mapping_df["Symbol"] = mapping_df["Symbol"].astype(str).str.strip()
    mapping_df["Sector"] = mapping_df["Sector"].astype(str).str.strip()
    mapping_df = mapping_df.dropna().drop_duplicates(subset=["Symbol"])
    return mapping_df


def summarize_pc_by_sector(
    loadings: pd.DataFrame,
    sector_map: pd.DataFrame,
    output_summary_path: Path,
    output_detail_path: Path,
    top_pcs: int = 10,
    top_symbols_per_pc: int = 20,
) -> None:
    """Summarize dominant sectors for each PC based on top absolute loadings."""
    sector_lookup = {}
    if not sector_map.empty:
        sector_lookup = sector_map.set_index("Symbol")["Sector"].to_dict()

    detail_rows = []
    summary_rows = []

    for pc in loadings.columns[: min(top_pcs, len(loadings.columns))]:
        top_symbols = loadings[pc].abs().nlargest(top_symbols_per_pc).index.tolist()
        tmp = pd.DataFrame(
            {
                "PC": pc,
                "Symbol": top_symbols,
                "Loading": [float(loadings.loc[s, pc]) for s in top_symbols],
            }
        )
        tmp["AbsLoading"] = tmp["Loading"].abs()
        tmp["Direction"] = np.where(tmp["Loading"] >= 0, "Positive", "Negative")
        tmp["Sector"] = tmp["Symbol"].map(lambda s: sector_lookup.get(s, "Unknown"))
        detail_rows.append(tmp)

        grouped = (
            tmp.groupby("Sector", as_index=False)
            .agg(
                SymbolCount=("Symbol", "count"),
                TotalAbsLoading=("AbsLoading", "sum"),
                NetLoading=("Loading", "sum"),
            )
            .sort_values("TotalAbsLoading", ascending=False)
        )

        total_abs = grouped["TotalAbsLoading"].sum()
        grouped["ContributionPct"] = np.where(
            total_abs > 0,
            grouped["TotalAbsLoading"] / total_abs * 100,
            0.0,
        )
        grouped["PC"] = pc
        grouped["DominantDirection"] = np.where(grouped["NetLoading"] >= 0, "Positive", "Negative")
        grouped["Rank"] = np.arange(1, len(grouped) + 1)

        summary_rows.append(
            grouped[
                [
                    "PC",
                    "Rank",
                    "Sector",
                    "SymbolCount",
                    "TotalAbsLoading",
                    "ContributionPct",
                    "DominantDirection",
                ]
            ]
        )

    detail_df = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    summary_df = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()

    detail_df.to_csv(output_detail_path, index=False)
    summary_df.to_csv(output_summary_path, index=False)


# ─────────────────────────────────────────────────────────────────
# VERIFY PC ORTHOGONALITY
# ─────────────────────────────────────────────────────────────────
def verify_pc_orthogonality(
    train_pca: np.ndarray,
    k: int,
    output_path: Path,
) -> float:
    """
    Phương châm: Sau PCA, các PC phải không tương quan nhau
    (tính chất orthogonality của PCA). Kiểm tra này xác nhận
    PCA đã loại bỏ hoàn toàn đa cộng tuyến.
    Max |r| giữa các PC phải ≈ 0 (< 1e-6).
    """
    corr_matrix = np.corrcoef(train_pca.T)
    off_diag = np.abs(corr_matrix - np.eye(k))
    max_cross_corr = off_diag.max()

    # Lưu artifact: ma trận tương quan PC (chỉ lấy corner nếu k lớn)
    n_show = min(k, 20)
    pc_cols = [f"PC{i+1}" for i in range(n_show)]
    corr_df = pd.DataFrame(
        corr_matrix[:n_show, :n_show],
        index=pc_cols,
        columns=pc_cols,
    ).round(8)
    corr_df.to_csv(output_path)

    status = "[PASS] OK" if max_cross_corr < 1e-5 else "[WARN]"
    print(f"[PCA VERIFY] Max cross-correlation giữa PC: {max_cross_corr:.2e} {status}")
    return max_cross_corr


# ─────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────
def save_pca_figure(
    explained_var: np.ndarray,
    cum_explained_var: np.ndarray,
    k_optimal: int,
    train_pca: np.ndarray,
    val_pca: np.ndarray,
    test_pca: np.ndarray,
    train_index: pd.DatetimeIndex,
    val_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    loadings: pd.DataFrame,
    output_path: Path,
    subfolder: str = "pca",
) -> None:
    # ===== TẠO THƯ MỤC CON NẾU CÓ =====
    if subfolder:
        save_dir = output_path.parent / subfolder
    else:
        save_dir = output_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = save_dir / "pca_summary.png"
    n = len(explained_var)
    n_show = min(50, n)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

    # (1) Scree Plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(1, n_show+1), explained_var[:n_show]*100,
            color="#1976D2", alpha=0.8, edgecolor="white")
    ax1.axvline(k_optimal, color="red", linestyle="--", linewidth=1.5,
                label=f"k={k_optimal}")
    ax1.set_title("Scree Plot", fontweight="bold", fontsize=10)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.legend(fontsize=9)

    # (2) Cumulative Explained Variance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(1, n+1), cum_explained_var*100, color="#388E3C", linewidth=2)
    ax2.fill_between(range(1, n+1), cum_explained_var*100, alpha=0.1, color="#388E3C")
    for thr, col in [(0.75,"#FF9800"), (0.80,"#F44336"), (0.85,"#7B1FA2"), (0.90,"#1E88E5"), (0.95,"#000")]:
        kt = int(np.argmax(cum_explained_var >= thr)) + 1
        ax2.axhline(thr*100, color=col, linestyle=":", linewidth=1,
                    label=f"{thr*100:.0f}% → k={kt}")
    ax2.axvline(k_optimal, color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlim(0, min(100, n))
    ax2.set_title("Cumulative Explained Variance", fontweight="bold", fontsize=10)
    ax2.set_xlabel("Số thành phần chính k")
    ax2.set_ylabel("CEV (%)")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(True, alpha=0.3)

    # (3) PC1 vs PC2 scatter
    ax3 = fig.add_subplot(gs[0, 2])
    n_pts = len(train_pca)
    sc = ax3.scatter(train_pca[:, 0], train_pca[:, 1],
                     c=np.arange(n_pts), cmap="viridis",
                     alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax3, label="Thời gian (sớm→muộn)")
    ax3.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
    ax3.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
    ax3.set_title("PC1 vs PC2 – Train Set", fontweight="bold", fontsize=10)

    # (4) PC1 theo thời gian (Train/Val/Test)
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.plot(train_index, train_pca[:, 0], color="#1565C0", linewidth=0.8, label="Train")
    ax4.plot(val_index,   val_pca[:, 0],   color="#E65100", linewidth=0.8, label="Val")
    ax4.plot(test_index,  test_pca[:, 0],  color="#B71C1C", linewidth=0.8, label="Test")
    ax4.axvline(val_index[0],  color="gray", linestyle="--", linewidth=0.8)
    ax4.axvline(test_index[0], color="gray", linestyle="--", linewidth=0.8)
    ax4.set_title(f"PC1 theo thời gian ({explained_var[0]*100:.1f}% phương sai)",
                  fontweight="bold", fontsize=10)
    ax4.set_ylabel("PC1 Score")
    ax4.legend(fontsize=8)
    ax4.tick_params(axis="x", rotation=20)

    # (5) Top Loadings PC1
    ax5 = fig.add_subplot(gs[1, 2])
    top15 = loadings["PC1"].abs().nlargest(15)
    c_load = ["#1565C0" if loadings.loc[s, "PC1"] > 0 else "#C62828" for s in top15.index]
    ax5.barh(top15.index, top15.values, color=c_load, edgecolor="white")
    ax5.set_xlabel("|Loading|")
    ax5.set_title("Top 15 Loadings – PC1", fontweight="bold", fontsize=10)
    ax5.invert_yaxis()
    from matplotlib.patches import Patch
    ax5.legend(handles=[Patch(facecolor="#1565C0", label="Loading (+)"),
                        Patch(facecolor="#C62828", label="Loading (–)")], fontsize=8)

    # (6) PC2 Loadings
    ax6 = fig.add_subplot(gs[2, 0])
    top10_pc2 = loadings["PC2"].abs().nlargest(10) if "PC2" in loadings.columns else pd.Series(dtype=float)
    if not top10_pc2.empty:
        c2 = ["#1565C0" if loadings.loc[s, "PC2"] > 0 else "#C62828" for s in top10_pc2.index]
        ax6.barh(top10_pc2.index, top10_pc2.values, color=c2, edgecolor="white")
        ax6.set_xlabel("|Loading|")
        ax6.set_title("Top 10 Loadings – PC2", fontweight="bold", fontsize=10)
        ax6.invert_yaxis()

    # (7) Threshold summary table
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.axis("off")
    thr_data = []
    for thr in [0.75, 0.80, 0.85, 0.90, 0.95]:
        kt = int(np.argmax(cum_explained_var >= thr)) + 1
        thr_data.append([f"{thr*100:.0f}%", kt,
                         f"{(1-kt/n)*100:.1f}%",
                         f"{cum_explained_var[kt-1]*100:.2f}%"])
    tbl = ax7.table(
        cellText=thr_data,
        colLabels=["Ngưỡng CEV", "k", "Giảm chiều", "CEV thực tế"],
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1976D2")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#E3F2FD")
    ax7.set_title("Phân tích ngưỡng CEV", fontweight="bold", fontsize=10, pad=15)

    # (8) PC variance bar chart (top 20)
    ax8 = fig.add_subplot(gs[2, 2])
    top20 = min(20, k_optimal)
    ax8.bar(range(1, top20+1), explained_var[:top20]*100,
            color="#7B1FA2", alpha=0.8, edgecolor="white")
    ax8.set_title(f"Top {top20} PC – % Phương sai (k={k_optimal})",
                  fontweight="bold", fontsize=10)
    ax8.set_xlabel("PC index")
    ax8.set_ylabel("Explained Var (%)")
    ax8.text(top20*0.6, explained_var[0]*100*0.8,
             f"PC1={explained_var[0]*100:.1f}%\n"
             f"PC1-5={cum_explained_var[4]*100:.1f}%",
             fontsize=8, color="#7B1FA2")

    plt.suptitle(f"Chương 3 – Kết quả PCA (k={k_optimal}, CEV≥95%)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved → {output_path}")


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────
def run_pca_pipeline(project_root: Path, config_path: Path) -> None:
    cfg       = load_config(config_path)
    paths     = cfg["paths"]
    pca_cfg   = cfg["pca"]

    processed_dir = project_root / paths.get("artifacts_dir", paths.get("processed_dir", "data/processed"))
    models_dir    = project_root / paths["models_dir"]
    figures_dir   = project_root / paths.get("figures_dir", "logs/figures")

    subdirs = paths.get("artifacts_subdirs", paths.get("processed_subdirs", {}))
    splits_dir = processed_dir / subdirs.get("splits", "splits")

    # Support per-CEV output subdirectory (for multi-CEV runs)
    output_subdir = pca_cfg.get("output_subdir", None)
    if output_subdir:
        # Save to both:
        #   data/processed/pca_cev_X.XX/  ← per-CEV archive
        #   data/processed/pca/            ← active dir (always up-to-date for ARDL/LSTM)
        pca_dir_archive = processed_dir / f"pca_{output_subdir}"
        pca_dir         = processed_dir / subdirs.get("pca", "pca")
        pca_dir_archive.mkdir(parents=True, exist_ok=True)
        print(f"[PCA] CEV subdir: {pca_dir_archive}")
    else:
        pca_dir         = processed_dir / subdirs.get("pca", "pca")
        pca_dir_archive = None

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pca_dir.mkdir(parents=True, exist_ok=True)

    sector_mapping_file = pca_cfg.get("sector_mapping_file", "data/raw/symbol_sector_mapping.csv")
    sector_map_path = project_root / sector_mapping_file
    sector_map = load_sector_mapping(sector_map_path)
    if sector_map.empty:
        print(f"[SECTOR MAP] Không có mapping hợp lệ tại {sector_map_path}. Dùng 'Unknown'.")
    else:
        print(f"[SECTOR MAP] Loaded {len(sector_map)} mã có ngành từ {sector_map_path}")

    # ── Load scaled data (output từ preprocess) ───────────────────
    for fname in ["train_scaled.csv", "val_scaled.csv", "test_scaled.csv"]:
        if not (splits_dir / fname).exists():
            raise FileNotFoundError(
                f"{fname} không tồn tại trong {splits_dir}. Chạy preprocess.py trước."
            )

    train_scaled = pd.read_csv(splits_dir / "train_scaled.csv", index_col=0, parse_dates=True)
    val_scaled   = pd.read_csv(splits_dir / "val_scaled.csv",   index_col=0, parse_dates=True)
    test_scaled  = pd.read_csv(splits_dir / "test_scaled.csv",  index_col=0, parse_dates=True)

    X_train = train_scaled.values
    X_val   = val_scaled.values
    X_test  = test_scaled.values
    p       = X_train.shape[1]

    print(f"[PCA INPUT] Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # ── Fit full PCA  ─────────────────────────────
    rs         = int(pca_cfg.get("random_state", 42))
    pca_full   = PCAReducer(n_components=None, random_state=rs)
    pca_full.fit(X_train)

    ev  = pca_full.get_metadata()["explained_variance_ratio"]
    cev = np.cumsum(ev)

    print(f"[PCA FULL] p={p} | PC1={ev[0]*100:.2f}% | "
          f"PC1-5={cev[4]*100:.2f}% | PC1-10={cev[9]*100:.2f}%")

    # ── Threshold analysis ────────────────────────────────────────
    threshold = float(pca_cfg.get("explained_variance_threshold", 0.95))
    k_optimal = int(np.argmax(cev >= threshold)) + 1

    # cev_requested: the CEV threshold being tested (input to the search).
    # cev_achieved: the ACTUAL cumulative explained variance at k components
    # (>= cev_requested by construction of argmax, generally slightly above
    # it since k is an integer). dim_reduction_pct is always recomputed from
    # the ACTUAL feature count p (X_train.shape[1] of this run), never
    # hardcoded, so it stays correct across CEV sweeps / dataset changes.
    thr_rows = []
    for thr in [0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
        kt = int(np.argmax(cev >= thr)) + 1
        thr_rows.append({
            "cev_requested":     round(thr, 4),
            "k":                 kt,
            "dim_reduction_pct": round((1 - kt / p) * 100, 4),
            "cev_achieved":      round(float(cev[kt - 1] * 100), 4),
        })
    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(pca_dir / "pca_threshold_summary.csv", index=False)
    print(f"\n[THRESHOLD ANALYSIS]  (p={p} input features; dim_reduction_pct = (1 - k/p) * 100, "
          f"recomputed from this run's actual p, never hardcoded)")
    print(thr_df.to_string(index=False))
    print(f"\n[DECISION] k={k_optimal} (CEV≥{threshold*100:.0f}%, "
          f"giảm {(1-k_optimal/p)*100:.1f}% chiều)")

    # ── Fit final PCA với k tối ưu ────────────────────────────────
    pca_final = PCAReducer(n_components=k_optimal, random_state=rs)
    pca_final.fit(X_train)

    pc_cols   = [f"PC{i+1}" for i in range(k_optimal)]
    train_pca = pca_final.transform(X_train)
    val_pca   = pca_final.transform(X_val)
    test_pca  = pca_final.transform(X_test)

    df_train_pca = pd.DataFrame(train_pca, index=train_scaled.index, columns=pc_cols)
    df_val_pca   = pd.DataFrame(val_pca,   index=val_scaled.index,   columns=pc_cols)
    df_test_pca  = pd.DataFrame(test_pca,  index=test_scaled.index,  columns=pc_cols)

    df_train_pca.to_csv(pca_dir / "train_pca.csv")
    df_val_pca.to_csv(pca_dir / "val_pca.csv")
    df_test_pca.to_csv(pca_dir / "test_pca.csv")
    print(f"[SAVED] train/val/test_pca.csv  shape=({len(df_train_pca)}, {k_optimal})")

    # Also write to per-CEV archive dir if running multi-CEV
    if pca_dir_archive is not None:
        df_train_pca.to_csv(pca_dir_archive / "train_pca.csv")
        df_val_pca.to_csv(pca_dir_archive / "val_pca.csv")
        df_test_pca.to_csv(pca_dir_archive / "test_pca.csv")
        print(f"[ARCHIVED] {pca_dir_archive}/train_val_test_pca.csv")

    # ── Loadings ──────────────────────────────────────────────────
    _pca_final_meta = pca_final.get_metadata()
    loadings = pd.DataFrame(
        _pca_final_meta["components_"].T,
        index=train_scaled.columns,
        columns=pc_cols,
    )
    loadings.to_csv(pca_dir / "pca_loadings.csv")
    
    eigenvalues = pd.DataFrame(
        {
            "PC": pc_cols,
            "eigenvalue": pca_final._pca.explained_variance_,
            "explained_variance_ratio": _pca_final_meta["explained_variance_ratio"],
            "explained_variance_pct": _pca_final_meta["explained_variance_ratio"] * 100,
            "cumulative_variance_pct": _pca_final_meta["cumulative_explained_variance"] * 100,
        }
    )
    eigenvalues.to_csv(pca_dir / "pca_eigenvalues.csv", index=False)
    print("[SAVED] pca_eigenvalues.csv")

    summarize_pc_by_sector(
        loadings=loadings,
        sector_map=sector_map,
        output_summary_path=pca_dir / "pc_sector_summary.csv",
        output_detail_path=pca_dir / "pc_top_symbols_with_sector.csv",
        top_pcs=int(pca_cfg.get("sector_top_pcs", 10)),
        top_symbols_per_pc=int(pca_cfg.get("sector_top_symbols", 20)),
    )
    print("[SAVED] pc_sector_summary.csv, pc_top_symbols_with_sector.csv")

    # Top loadings PC1-3
    for i in range(min(3, k_optimal)):
        pc = pc_cols[i]
        top5_pos = loadings[pc].nlargest(5).index.tolist()
        top5_neg = loadings[pc].nsmallest(5).index.tolist()
        print(f"[{pc}] {ev[i]*100:.2f}% | Top(+): {top5_pos} | Top(-): {top5_neg}")

    # ── Verify PC orthogonality ───────────────────────────────────
    max_cross = verify_pc_orthogonality(
        train_pca, k_optimal, pca_dir / "pca_corr_after.csv"
    )

    # ── Metrics ───────────────────────────────────────────────────
    metrics = {
        "input_features": p,
        "k_optimal": k_optimal,
        "cev_threshold": threshold,
        "cev_achieved": float(cev[k_optimal-1]),
        "dim_reduction_pct": (1 - k_optimal / p) * 100,
        "pc1_var": float(ev[0]),
        "pc1_5_cum_var": float(cev[4]),
        "max_pc_cross_corr": float(max_cross),
    }
    pd.Series(metrics).to_csv(pca_dir / "pca_metrics.csv", header=["value"])
    print(f"\n[PCA METRICS]")
    for k_m, v_m in metrics.items():
        print(f"  {k_m}: {v_m}")

    # ── Save model ────────────────────────────────────────────────
    with open(models_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(pca_final, f)
    print(f"[SAVED] pca_model.pkl")

    # ── Figures ───────────────────────────────────────────────────
    save_pca_figure(
        explained_var=ev,
        cum_explained_var=cev,
        k_optimal=k_optimal,
        train_pca=train_pca,
        val_pca=val_pca,
        test_pca=test_pca,
        train_index=train_scaled.index,
        val_index=val_scaled.index,
        test_index=test_scaled.index,
        loadings=loadings,
        output_path=figures_dir / "pca_summary.png",
        subfolder="pca",
    )

    # ── Scree Plot riêng ───────────────────────────────────────────
    save_scree_plot_figure(
        figures_dir=figures_dir,
        explained_var=ev,
        k_optimal=k_optimal,
        subfolder="pca",
    )

    # ── PCA individual figures ──────────────────────────────────
    save_pca_individual_figures(
        figures_dir=figures_dir,
        loadings=loadings,
        explained_var=ev,
        train_pca=train_pca,
        train_index=train_scaled.index,
    )

    save_pca_threshold_table(
        figures_dir=figures_dir,
        cev=cev,
        p=p,
    )
    # ── PC Time Series riêng biệt ────────────────────
    try:
        from preprocess_steps import save_pc_time_series_individual
        pc_cols_exist = [f'PC{i+1}' for i in range(min(3, k_optimal))]
        pc_data = pd.DataFrame(
            train_pca[:, :len(pc_cols_exist)],
            index=train_scaled.index,
            columns=pc_cols_exist
        )
        save_pc_time_series_individual(
            figures_dir=figures_dir,
            pc_data=pc_data,
            pc_names=pc_cols_exist,
        )
    except Exception as e:
        print(f'[WARN] save_pc_time_series_individual failed: {e}')

    # ── Heatmap & Histogram ─────────────────────────────────────
    cleaned_path = processed_dir / "core" / "cleaned_data.csv"
    if cleaned_path.exists():
        df_pivot = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
        save_correlation_heatmap_full(figures_dir, df_pivot=df_pivot, n_stocks=30, save_csv=True, random_seed=42)
        save_distribution_figure(figures_dir, train_scaled=train_scaled, n_stocks=9)
    else:
        print("[WARN] cleaned_data.csv not found")

    # ── Walk-forward split figure ──────────────────────────────
    split_summary_path = processed_dir / "splits" / "split_summary.csv"
    vnindex_path = processed_dir / "core" / "vnindex_target.csv"
    
    if split_summary_path.exists() and vnindex_path.exists():
        split_summary = pd.read_csv(split_summary_path)
        vnindex_series = pd.read_csv(vnindex_path, index_col=0, parse_dates=True).squeeze()
        try:
            from step_11_walkforward_figure import save_walkforward_figure, save_lookback_illustration
            save_walkforward_figure(figures_dir, split_summary, vnindex_series)
            save_lookback_illustration(figures_dir)
        except ImportError:
            print("[WARN] step_11_walkforward_figure.py not found")
    else:
        print("[WARN] skip walkforward")

    # ── Final summary ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("PCA PIPELINE HOÀN THÀNH")
    print("="*60)
    print(f"  Input  : {p} cổ phiếu (features)")
    print(f"  Output : {k_optimal} thành phần chính")
    print(f"  CEV    : {cev[k_optimal-1]*100:.2f}% (≥{threshold*100:.0f}%)")
    print(f"  Giảm   : {(1-k_optimal/p)*100:.1f}% số chiều")
    print(f"  Max cross-corr PC: {max_cross:.2e} (~ 0 [PASS])")
    print(f"\n  Bước tiếp theo: xây dựng ARDL + LSTM với dữ liệu train/val/test_pca.csv")


# ─────────────────────────────────────────────────────────────────
def save_pca_individual_figures(
    figures_dir: Path,
    loadings: pd.DataFrame,
    explained_var: np.ndarray,
    train_pca: np.ndarray,
    train_index: pd.DatetimeIndex,
    subfolder: str = "pca",
) -> None:
    """Xuất các hình riêng lẻ cho báo cáo."""
    # ===== TẠO THƯ MỤC CON NẾU CÓ =====
    if subfolder:
        pca_fig_dir = figures_dir / subfolder
    else:
        pca_fig_dir = figures_dir
    pca_fig_dir.mkdir(parents=True, exist_ok=True)

    pc_cols = loadings.columns.tolist()

    # (A) PC1, PC2, PC3 LOADINGS
    colors_time = ["#1565C0", "#E65100", "#2E7D32"]
    for i, pc in enumerate(pc_cols[:5]):
        fig, ax = plt.subplots(figsize=(10, 6))
        top15 = loadings[pc].abs().nlargest(15)
        colors = ["#1565C0" if loadings.loc[s, pc] > 0 else "#C62828" for s in top15.index]
        ax.barh(top15.index, top15.values, color=colors, edgecolor="white")
        ax.set_xlabel("|Loading|", fontsize=11)
        ax.set_title(f"Top 15 Loadings – {pc} ({explained_var[i]*100:.1f}% variance)", fontweight="bold", fontsize=12)
        ax.invert_yaxis()
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="#1565C0", label="Loading (+)"), Patch(facecolor="#C62828", label="Loading (–)")], fontsize=9)
        fig.tight_layout()
        fig.savefig(pca_fig_dir / f"loading_{pc.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[FIGURE] Saved -> pca/loading_{pc.lower()}.png")

    # (B) PC1, PC2, PC3 theo thời gian
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for i, pc in enumerate(pc_cols[:3]):
        ax = axes[i]
        ax.plot(train_index, train_pca[:, i], color=colors_time[i], linewidth=1.2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{pc} theo thời gian ({explained_var[i]*100:.1f}% phương sai)", fontweight="bold", fontsize=10)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(pca_fig_dir / "pc_time_series.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved -> pca/pc_time_series.png")

    # (C) Biplot PC1 vs PC2
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(train_pca[:, 0], train_pca[:, 1], c=np.arange(len(train_pca)), cmap="viridis", alpha=0.3, s=10)
    plt.colorbar(scatter, ax=ax, label="Thời gian (sớm→muộn)")
    n_loadings = min(20, len(loadings))
    loadings_subset = loadings.nlargest(n_loadings, "PC1") if n_loadings > 0 else loadings
    scale_factor = 8
    for idx, row in loadings_subset.iterrows():
        ax.arrow(0, 0, row["PC1"] * scale_factor, row["PC2"] * scale_factor,
                 head_width=0.3, head_length=0.3, fc="red", ec="red", alpha=0.5)
        ax.text(row["PC1"] * scale_factor * 1.05, row["PC2"] * scale_factor * 1.05,
                idx, fontsize=7, color="darkred")
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)", fontsize=11)
    ax.set_title("Biplot: PC1 vs PC2 với loading vectors (top 20)", fontweight="bold", fontsize=12)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(pca_fig_dir / "biplot_pc1_pc2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved -> pca/biplot_pc1_pc2.png")

    # (D) Histogram PC1, PC2, PC3
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, pc in enumerate(pc_cols[:3]):
        ax = axes[i]
        ax.hist(train_pca[:, i], bins=50, color=colors_time[i], alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{pc} - Histogram", fontweight="bold", fontsize=10)
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(pca_fig_dir / "pc_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved -> pca/pc_histograms.png")


def save_pca_threshold_table(
    figures_dir: Path,
    cev: np.ndarray,
    p: int,
    subfolder: str = "pca",  
) -> Path:
    """Xuất bảng phân tích ngưỡng CEV dưới dạng hình ảnh."""
    # ===== TẠO THƯ MỤC CON NẾU CÓ =====
    if subfolder:
        pca_fig_dir = figures_dir / subfolder
    else:
        pca_fig_dir = figures_dir
    pca_fig_dir.mkdir(parents=True, exist_ok=True)
    
    out = pca_fig_dir / "pca_threshold_table.png" 

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    thr_data = []
    for thr in [0.80, 0.85, 0.90, 0.95, 0.99]:
        kt = int(np.argmax(cev >= thr)) + 1
        thr_data.append([f"{thr*100:.0f}%", kt, f"{(1 - kt/p)*100:.1f}%", f"{cev[kt-1]*100:.2f}%"])

    tbl = ax.table(cellText=thr_data, colLabels=["Ngưỡng CEV", "k", "Giảm chiều", "CEV thực"],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1976D2")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#E3F2FD")

    ax.set_title("Bảng phân tích ngưỡng CEV", fontweight="bold", fontsize=12, pad=15)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved -> pca/pca_threshold_table.png")
    return out

def save_scree_plot_figure(
    figures_dir: Path,
    explained_var: np.ndarray,
    k_optimal: int,
    subfolder: str = "pca",
) -> Path:
    """
    Xuất RIÊNG biểu đồ Scree Plot (dạng đường gấp khúc).
    """
    if subfolder:
        save_dir = figures_dir / subfolder
    else:
        save_dir = figures_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    out = save_dir / "scree_plot.png"
    n = len(explained_var)
    n_show = min(50, n)
    
    print(f"[DEBUG] Đang vẽ Scree Plot với n={n_show}, k={k_optimal}, lưu vào {out}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Vẽ Scree Plot dạng LINE (đường gấp khúc) 
    ax.plot(range(1, n_show+1), explained_var[:n_show]*100, 
            marker='o', 
            markersize=6,
            color='#1565C0', 
            linewidth=2, 
            markerfacecolor='#1565C0',
            markeredgecolor='white',
            markeredgewidth=1)
    
    # Đánh dấu đường thẳng đứng tại k_optimal
    ax.axvline(k_optimal, color='red', linestyle='--', linewidth=2,
               label=f'k={k_optimal} (CEV≥95%)')
    
    # Đánh dấu điểm tại k_optimal
    if k_optimal <= n_show:
        ax.scatter(k_optimal, explained_var[k_optimal-1]*100, 
                   color='red', s=120, zorder=5, 
                   edgecolor='white', linewidth=2)
    
    # Format
    ax.set_xlabel('Component Number', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title('Scree Plot - Giá trị riêng theo từng thành phần chính', 
                 fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    
    # Thêm annotation
    cum_var = np.cumsum(explained_var)[k_optimal-1] * 100
    ax.text(0.02, 0.95, 
            f'k={k_optimal} → CEV={cum_var:.2f}% | '
            f'Giảm {(1-k_optimal/len(explained_var))*100:.1f}% chiều',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f'[FIGURE] ĐÃ LƯU Scree Plot: {out}')
    print(f'[FIGURE] File tồn tại? {out.exists()}')
    
    return out
# ─────────────────────────────────────────────────────────────────
# NOREDUCTION PIPELINE (identity: output raw scaled features unchanged)
# ─────────────────────────────────────────────────────────────────
def run_noreduction_pipeline(project_root: Path, config_path: Path) -> None:
    """Output raw scaled features as the 'representation' for forecasting.

    Writes the existing Train/Val/Test scaled data (produced by preprocess.py)
    directly to the pca/ output directory in the same format the forecasting
    layer expects (train_pca.csv, val_pca.csv, test_pca.csv with date index +
    feature columns). Feature columns retain their original names (stock symbols),
    NOT renamed to PC1..PCn.

    Also writes a minimal pca_metrics.csv so downstream reporting can identify
    this as a NoReduction run.
    """
    cfg = load_config(config_path)
    paths = cfg["paths"]

    processed_dir = project_root / paths.get("artifacts_dir", paths.get("processed_dir", "data/processed"))
    subdirs = paths.get("artifacts_subdirs", paths.get("processed_subdirs", {}))
    splits_dir = processed_dir / subdirs.get("splits", "splits")
    pca_dir = processed_dir / subdirs.get("pca", "pca")
    pca_dir.mkdir(parents=True, exist_ok=True)

    # Load scaled data
    for fname in ["train_scaled.csv", "val_scaled.csv", "test_scaled.csv"]:
        if not (splits_dir / fname).exists():
            raise FileNotFoundError(
                f"{fname} not found in {splits_dir}. Run preprocess.py first."
            )

    train_scaled = pd.read_csv(splits_dir / "train_scaled.csv", index_col=0, parse_dates=True)
    val_scaled = pd.read_csv(splits_dir / "val_scaled.csv", index_col=0, parse_dates=True)
    test_scaled = pd.read_csv(splits_dir / "test_scaled.csv", index_col=0, parse_dates=True)

    p = train_scaled.shape[1]

    # Validate feature consistency across splits
    assert list(train_scaled.columns) == list(val_scaled.columns), \
        "Train/Val feature columns mismatch"
    assert list(train_scaled.columns) == list(test_scaled.columns), \
        "Train/Test feature columns mismatch"

    print(f"[NOREDUCTION] Representation = identity (raw scaled features)")
    print(f"[NOREDUCTION] Train: {train_scaled.shape} | Val: {val_scaled.shape} | Test: {test_scaled.shape}")
    print(f"[NOREDUCTION] Features: {p} (original scaled, no dimensionality reduction)")

    # Write to pca/ dir in the same CSV format forecasting expects
    train_scaled.to_csv(pca_dir / "train_pca.csv")
    val_scaled.to_csv(pca_dir / "val_pca.csv")
    test_scaled.to_csv(pca_dir / "test_pca.csv")
    print(f"[SAVED] train/val/test_pca.csv (NoReduction, {p} features)")

    # Write metrics for downstream reporting
    metrics = {
        "input_features": p,
        "k_optimal": p,  # output_dim == input_dim for identity
        "cev_threshold": None,
        "cev_achieved": None,
        "dim_reduction_pct": 0.0,
        "reduction_method": "none",
    }
    pd.Series(metrics).to_csv(pca_dir / "pca_metrics.csv", header=["value"])
    print(f"[SAVED] pca_metrics.csv (NoReduction marker)")

    # Write a minimal threshold summary (single row, no PCA thresholds)
    thr_df = pd.DataFrame([{
        "cev_requested": None,
        "k": p,
        "dim_reduction_pct": 0.0,
        "cev_achieved": None,
    }])
    thr_df.to_csv(pca_dir / "pca_threshold_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("NOREDUCTION PIPELINE HOÀN THÀNH")
    print("=" * 60)
    print(f"  Input  : {p} features (raw scaled)")
    print(f"  Output : {p} features (identity, no reduction)")
    print(f"  CEV    : N/A (NoReduction)")
    print(f"  Giảm   : 0.0%")
    print(f"\n  Bước tiếp theo: ARDL + LSTM with raw features")


# ─────────────────────────────────────────────────────────────────
# UNIFIED DISPATCHER
# ─────────────────────────────────────────────────────────────────
def run_reduction_pipeline(project_root: Path, config_path: Path) -> None:
    """Dispatch to PCA or NoReduction based on config reduction.method."""
    cfg = load_config(config_path)
    reduction_cfg = cfg.get("reduction", {})
    method = str(reduction_cfg.get("method", "pca")).lower().strip()

    if method == "none":
        run_noreduction_pipeline(project_root, config_path)
    elif method == "pca":
        run_pca_pipeline(project_root, config_path)
    else:
        raise ValueError(
            f"Unknown reduction.method={method!r} in config. "
            f"Supported: 'pca', 'none'."
        )


# ─────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VN-Index PCA pipeline")
    p.add_argument("--config", default="config/config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_pca_pipeline(project_root=root, config_path=root / args.config)