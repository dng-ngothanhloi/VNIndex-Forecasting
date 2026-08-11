from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_preprocess_figure(
    figures_dir: Path,
    vnindex_series: pd.Series,
    df_pivot: pd.DataFrame,
    train_data: pd.DataFrame,
    train_scaled: pd.DataFrame,
    stationarity_df: pd.DataFrame,
    corr_summary: pd.DataFrame,
    n_train: int,
    n_val: int,
    subfolder: str = "pca",
) -> Path:
    """
    Tổng hợp 6 biểu đồ thành 1 figure để gắn vào báo cáo.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== TẠO THƯ MỤC CON NẾU CÓ =====
    if subfolder:
        save_dir = figures_dir / subfolder
    else:
        save_dir = figures_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    out = save_dir / "preprocess_summary.png"

    sym = "ACB" if "ACB" in train_data.columns else train_data.columns[0]
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

    # (1) VN-Index với splits
    ax1 = fig.add_subplot(gs[0, :2])
    idx = vnindex_series.index
    ax1.plot(idx, vnindex_series.values, color="#1565C0", linewidth=0.9)
    ax1.fill_between(idx, vnindex_series.min(), vnindex_series.values, alpha=0.07, color="#1565C0")
    if n_train < len(vnindex_series):
        ax1.axvline(idx[n_train - 1], color="orange", linestyle="--", linewidth=1.2, label="Train/Val split")
    if n_train + n_val < len(vnindex_series):
        ax1.axvline(idx[n_train + n_val - 1], color="red", linestyle="--", linewidth=1.2, label="Val/Test split")
    ax1.set_title("Chuỗi VN-Index (2020-2025) với phân vùng Train / Val / Test", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Điểm VN-Index")
    ax1.legend(fontsize=8)
    ax1.tick_params(axis="x", rotation=20)

    # (2) Phân bố missing
    ax2 = fig.add_subplot(gs[0, 2])
    missing_ratio = df_pivot.isnull().mean()
    bins = [
        ("0%", missing_ratio == 0),
        ("0-5%", (missing_ratio > 0) & (missing_ratio <= 0.05)),
        ("5-10%", (missing_ratio > 0.05) & (missing_ratio <= 0.10)),
        ("10-20%", (missing_ratio > 0.10) & (missing_ratio <= 0.20)),
        (">20%", missing_ratio > 0.20),
    ]
    labels = [b[0] for b in bins]
    vals = [b[1].sum() for b in bins]
    colors = ["#2E7D32", "#66BB6A", "#FFA726", "#EF5350", "#B71C1C"]
    ax2.bar(labels, vals, color=colors, edgecolor="white")
    ax2.set_title("Phân bố tỷ lệ thiếu theo mã", fontweight="bold", fontsize=10)
    ax2.set_ylabel("Số mã")

    # (3) Phân phối gốc
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(train_data[sym].dropna(), bins=40, color="#1976D2", alpha=0.8, edgecolor="white")
    ax3.set_title(f"Phân phối gốc - {sym}", fontweight="bold", fontsize=10)
    ax3.set_xlabel("Giá đóng cửa (VND)")

    # (4) Phân phối sau Z-score
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(train_scaled[sym].dropna(), bins=40, color="#388E3C", alpha=0.8, edgecolor="white")
    ax4.set_title(f"Sau Z-score - {sym}", fontweight="bold", fontsize=10)
    ax4.set_xlabel("z-value")
    ax4.axvline(0, color="red", linestyle="--", linewidth=1)

    # (5) ADF p-value
    ax5 = fig.add_subplot(gs[1, 2])
    if not stationarity_df.empty:
        pvals = stationarity_df["ADF_p"].fillna(1.0).values
        syms = stationarity_df["Symbol"].tolist()
        colors_bar = ["#2E7D32" if p < 0.05 else "#C62828" for p in pvals]
        ax5.bar(range(len(pvals)), pvals, color=colors_bar, edgecolor="white")
        ax5.axhline(0.05, color="black", linestyle="--", linewidth=1)
        ax5.set_xticks(range(len(pvals)))
        ax5.set_xticklabels(syms, rotation=90, fontsize=6)
        ax5.set_title("ADF p-value (30 mã đại diện)", fontweight="bold", fontsize=10)
        ax5.set_ylabel("p-value")

        from matplotlib.patches import Patch
        ax5.legend(
            handles=[
                Patch(facecolor="#2E7D32", label="Dừng p<0.05"),
                Patch(facecolor="#C62828", label="Không dừng"),
            ],
            fontsize=7,
        )

    # (6) Heatmap tương quan 
    ax6 = fig.add_subplot(gs[2, :])
    
    n_show = min(30, df_pivot.shape[1])
    all_stocks = df_pivot.columns.tolist()
    rng = np.random.RandomState(42)
    shuffled_stocks = rng.permutation(all_stocks).tolist()
    selected_stocks = shuffled_stocks[:n_show]
    sub_corr = df_pivot[selected_stocks].corr()
    
    # ===== IN STATS  =====
    # Lấy tam giác trên, bỏ đường chéo
    upper_tri = sub_corr.where(np.triu(np.ones(sub_corr.shape), k=1).astype(bool))
    corr_vals = upper_tri.stack()
    abs_corr_vals = corr_vals.abs()
    
    # Tính số cặp KHÔNG LẶP và KHÔNG ĐẾM CHÍNH NÓ
    total_pairs = n_show * (n_show - 1) // 2 
    
    n_ge_07 = (abs_corr_vals >= 0.7).sum()
    n_ge_08 = (abs_corr_vals >= 0.8).sum()
    n_ge_09 = (abs_corr_vals >= 0.9).sum()
    mean_abs_corr = abs_corr_vals.mean()
    
    print("\n" + "=" * 60)
    print(f" THỐNG KÊ TƯƠNG QUAN ({n_show} cổ phiếu ngẫu nhiên)")
    print("=" * 60)
    print(f"  Tổng số cặp              : {total_pairs:,} (={n_show}*{n_show-1}/2)")  # ← 435
    print(f"  Cặp |r| ≥ 0.7           : {n_ge_07:,} ({n_ge_07/total_pairs*100:.1f}%)")
    print(f"  Cặp |r| ≥ 0.8           : {n_ge_08:,} ({n_ge_08/total_pairs*100:.1f}%)")
    print(f"  Cặp |r| ≥ 0.9           : {n_ge_09:,} ({n_ge_09/total_pairs*100:.1f}%)")
    print(f"  Trung bình |r|          : {mean_abs_corr:.4f}")
    
    top5 = abs_corr_vals.nlargest(5)
    if len(top5) > 0:
        print(f"\n   Top 5 cặp tương quan cao nhất:")
        for (s1, s2), val in top5.items():
            print(f"     {s1} - {s2}: {val:.4f}")
    print("=" * 60 + "\n")
    
    # ===== VẼ HEATMAP =====
    # mask = np.triu(np.ones_like(sub_corr, dtype=bool), k=1)  
    sns.heatmap(
        sub_corr,
        mask=None,  # ← KHÔNG CHE
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax6,
        cbar_kws={"shrink": 0.5, "label": "Correlation"},
        xticklabels=False,
        yticklabels=False,
        linewidths=0,
    )
    ax6.set_title(
        f"Ma trận tương quan Pearson - {n_show} mã ngẫu nhiên (xác nhận đa cộng tuyến cao trước PCA)",
        fontweight="bold",
        fontsize=10,
    )

# ================================================================
# Heatmap  xuất CSV
# ================================================================
def save_correlation_heatmap_full(
    figures_dir: Path,
    df_pivot: pd.DataFrame,
    n_stocks: int = 30,
    subfolder: str = "pca",
    save_csv: bool = True,
    random_seed: int = 42,
) -> Path:
    """
    Vẽ heatmap ma trận tương quan và xuất file CSV.

    Parameters
    ----------
    figures_dir : Path
        Thư mục lưu ảnh
    df_pivot : pd.DataFrame
        Dữ liệu dạng wide (ngày × mã cổ phiếu)
    n_stocks : int
        Số lượng cổ phiếu để hiển thị
    subfolder : str
        Thư mục con trong figures_dir
    save_csv : bool
        Có xuất file CSV hay không

    Returns
    -------
    Path
        Đường dẫn đến file ảnh đã lưu
    """
    # ===== TẠO THƯ MỤC =====
    if subfolder:
        save_dir = figures_dir / subfolder
    else:
        save_dir = figures_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    out_img = save_dir / "correlation_heatmap_full.png"
    out_csv = save_dir / "correlation_matrix_full.csv"

    # ===== LẤY n_stocks ngẫu nhiên =====
    n_show = min(n_stocks, df_pivot.shape[1])

    all_stocks = df_pivot.columns.tolist()
    rng = np.random.RandomState(random_seed)
    shuffled_stocks = rng.permutation(all_stocks).tolist()
    selected_stocks = shuffled_stocks[:n_show]
    sub_corr = df_pivot[selected_stocks].corr()
    
    # ===== XUẤT CSV =====
    if save_csv:
        sub_corr.to_csv(out_csv)
        print(f"[CSV] Saved correlation matrix -> {out_csv}")

    # ===== VẼ HEATMAP =====
    fig, ax = plt.subplots(figsize=(16, 14))

    sns.heatmap(
        sub_corr,
        mask=None,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={
            "shrink": 0.6,
            "label": "Pearson Correlation",
            "ticks": [-1.0, -0.5, 0, 0.5, 1.0]
        },
        xticklabels=selected_stocks,
        yticklabels=selected_stocks,
        linewidths=0.5,
        linecolor='white',
        annot=True,
        fmt='.2f',
        annot_kws={'size': 7},
    )

    ax.set_xticklabels(selected_stocks, rotation=90, fontsize=8)
    ax.set_yticklabels(selected_stocks, rotation=0, fontsize=8)

    ax.set_title(
        f"Ma trận tương quan Pearson - {n_show} cổ phiếu (thể hiện đa cộng tuyến trước PCA)",
        fontweight="bold",
        fontsize=14,
        pad=20,
    )

    # ===== THỐNG KÊ TÓM TẮT =====
    # Tính tỷ lệ cặp có tương quan > 0.7
    upper_tri = sub_corr.where(np.triu(np.ones(sub_corr.shape), k=1).astype(bool))
    high_corr_pairs = (upper_tri.abs() > 0.7).sum().sum()
    total_pairs = n_show * (n_show - 1) // 2
    high_corr_pct = high_corr_pairs / total_pairs * 100 if total_pairs > 0 else 0

    ax.text(
        0.5, -0.08,
        f"Màu đậm = |r| > 0.70 (đa cộng tuyến) | "
        f"Cặp có |r| > 0.70: {high_corr_pairs}/{total_pairs} ({high_corr_pct:.1f}%)",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(out_img, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved -> {out_img}")

    return out_img

# ================================================================
# Histogram Z-score
# ================================================================
def save_distribution_figure(
    figures_dir: Path,
    train_scaled: pd.DataFrame,
    n_stocks: int = 9,
    subfolder: str = "pca",  
    random_seed: int = 42,
) -> Path:
    """Vẽ histogram phân phối của 9 cổ phiếu sau Z-score."""
    # ===== TẠO THƯ MỤC CON NẾU CÓ =====
    if subfolder:
        save_dir = figures_dir / subfolder
    else:
        save_dir = figures_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    out = save_dir / "distribution_zscore.png"

    n_show = min(n_stocks, train_scaled.shape[1])
    all_cols = train_scaled.columns.tolist()
    rng = np.random.RandomState(random_seed)
    shuffled_cols = rng.permutation(all_cols).tolist()
    selected_cols = shuffled_cols[:n_show]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    for i, col in enumerate(selected_cols):
        ax = axes[i]
        ax.hist(train_scaled[col].dropna(), bins=40, color="#1976D2", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("z-score", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)

    for j in range(len(selected_cols), 9):
        axes[j].axis("off")

    plt.suptitle(f"Phân phối Z-score của {n_show} cổ phiếu ngẫu nhiên", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved -> {out}")
    return out

# ================================================================
# BOXPLOT IQR COMPARISON 
# ================================================================
def save_boxplot_iqr_comparison(
    figures_dir: Path,
    stocks_df: pd.DataFrame,
    stocks_clean: pd.DataFrame,
    symbol_col: str = "Symbol",
    close_col: str = "Close",
    n_stocks: int = 5,
    random_seed: int = 42,
    subfolder: str = "pca",
) -> Path:
    """
    Vẽ boxplot so sánh trước và sau IQR outlier removal.
    """
    if subfolder:
        save_dir = figures_dir / subfolder
    else:
        save_dir = figures_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    out = save_dir / "boxplot_iqr_comparison.png"
    
    # Chọn ngẫu nhiên n_stocks mã
    all_symbols = stocks_df[symbol_col].unique()
    rng = np.random.RandomState(random_seed)
    selected = rng.choice(all_symbols, size=min(n_stocks, len(all_symbols)), replace=False)
    
    fig, axes = plt.subplots(n_stocks, 2, figsize=(14, 4*n_stocks))
    if n_stocks == 1:
        axes = axes.reshape(1, 2)
    
    for i, sym in enumerate(selected):
        # Dữ liệu trước
        before = stocks_df[stocks_df[symbol_col] == sym][close_col].values
        # Dữ liệu sau
        after = stocks_clean[stocks_clean[symbol_col] == sym][close_col].values
        
        # Boxplot trước
        ax_before = axes[i, 0]
        bp1 = ax_before.boxplot(before, vert=True, patch_artist=True)
        bp1['boxes'][0].set_facecolor('#FF6B6B')
        ax_before.set_title(f'{sym} - Trước IQR', fontweight='bold')
        ax_before.set_ylabel('Giá đóng cửa (VND)')
        
        # Boxplot sau
        ax_after = axes[i, 1]
        bp2 = ax_after.boxplot(after, vert=True, patch_artist=True)
        bp2['boxes'][0].set_facecolor('#51CF66')
        ax_after.set_title(f'{sym} - Sau IQR', fontweight='bold')
        ax_after.set_ylabel('Giá đóng cửa (VND)')
    
    plt.suptitle('So sánh phân phối giá trước và sau xử lý ngoại lai IQR', 
                 fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[FIGURE] Saved -> {out}')
    return out


# ================================================================
# MISSING HEATMAP 
# ================================================================
def save_missing_heatmap(
    figures_dir: Path,
    df_pivot: pd.DataFrame,
    subfolder: str = "pca",
    max_stocks: int = 100,
) -> Path:
    """
    Vẽ heatmap missing values: trục x = thời gian, trục y = mã cổ phiếu.
    """
    if subfolder:
        save_dir = figures_dir / subfolder
    else:
        save_dir = figures_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    out = save_dir / "missing_heatmap.png"
    
    # Tạo ma trận missing (1 = missing, 0 = có dữ liệu)
    missing_matrix = df_pivot.isnull().astype(int)
    
    # Chỉ hiển thị tối đa max_stocks mã để heatmap dễ nhìn
    n_show = min(max_stocks, missing_matrix.shape[1])
    missing_subset = missing_matrix.iloc[:, :n_show]
    
    # Sắp xếp theo tỷ lệ missing giảm dần
    missing_ratio = missing_subset.mean(axis=0)
    missing_subset = missing_subset.iloc[:, missing_ratio.argsort()[::-1]]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    im = ax.imshow(missing_subset.T, aspect='auto', cmap='RdYlBu_r',
                   interpolation='nearest')
    
    ax.set_xlabel('Thời gian (ngày giao dịch)', fontsize=12)
    ax.set_ylabel('Mã cổ phiếu', fontsize=12)
    ax.set_title('Heatmap dữ liệu thiếu theo thời gian và mã cổ phiếu\n(Đỏ = thiếu, Xanh = có dữ liệu)',
                 fontweight='bold', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Missing (1=thiếu, 0=có dữ liệu)', fontsize=10)
    
    total_missing = missing_matrix.sum().sum()
    total_cells = missing_matrix.shape[0] * missing_matrix.shape[1]
    pct_missing = total_missing / total_cells * 100
    
    ax.text(0.5, -0.08, 
            f'Tổng dữ liệu thiếu: {total_missing:,}/{total_cells:,} ({pct_missing:.2f}%)',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')
    
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[FIGURE] Saved -> {out}')
    return out


# ================================================================
# PC TIME SERIES INDIVIDUAL 
# ================================================================
def save_pc_time_series_individual(
    figures_dir: Path,
    pc_data: pd.DataFrame,
    pc_names: list = None,
    subfolder: str = "pca",
) -> list:
    """
    Vẽ 3 line chart riêng biệt cho PC1, PC2, PC3 theo thời gian.
    """
    if pc_names is None:
        pc_names = ['PC1', 'PC2', 'PC3']
    
    if subfolder:
        save_dir = figures_dir / subfolder
    else:
        save_dir = figures_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = []
    colors = ['#1565C0', '#E65100', '#2E7D32']
    
    for i, pc in enumerate(pc_names):
        if pc not in pc_data.columns:
            continue
            
        out = save_dir / f'pc_time_{pc.lower()}.png'
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(pc_data.index, pc_data[pc].values, 
                color=colors[i % len(colors)], linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.fill_between(pc_data.index, 0, pc_data[pc].values,
                        alpha=0.1, color=colors[i % len(colors)])
        
        ax.set_title(f'{pc} theo thời gian 2020-2025', 
                     fontweight='bold', fontsize=14)
        ax.set_xlabel('Thời gian', fontsize=12)
        ax.set_ylabel('PC Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        stats_text = (f'Mean: {pc_data[pc].mean():.4f} | '
                      f'Std: {pc_data[pc].std():.4f} | '
                      f'Min: {pc_data[pc].min():.4f} | '
                      f'Max: {pc_data[pc].max():.4f}')
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        outputs.append(out)
        print(f'[FIGURE] Saved -> {out}')
    
    return outputs