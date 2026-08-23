"""
analyze_pc_attribution.py — Exploratory attribution analysis for the LSTM
forecaster: which principal components drive the forecast, and can that be
mapped back to the original stocks?

STATUS: EXPLORATORY. This is not part of the O1–O4 / RQ1–RQ3 evidence chain.
Read `notebooks/Experiment_Results_Analysis.md`, Appendix A, before using any
number produced here. Two limitations are structural, not fixable by tuning
this script:

  1. The LSTM family in the current run carries a known early-stopping defect
     (17/60 tuning configs collapsed to an untrained epoch-1 model). Only
     representations with a 0/10 collapse rate are analysed here; the others
     are refused outright rather than reported with a caveat.
  2. Each analysed model is a SINGLE seed (42). The project's multi-seed
     evidence shows Test RMSE varies materially across seeds, so the
     attribution profile below is one draw, not an estimate of a population.
     A publishable version must average over seeds 42/52/62/72/82.

Method
------
Permutation importance on the input channels of the deployed (final-refit)
model, reconstructed from the persisted bundle so nothing is retrained:

    input tensor  : (n_samples, lookback, k + 1)
    channels      : PC1..PCk, plus one channel of lagged VNINDEX
    importance    : increase in Test RMSE when one channel is shuffled across
                    the sample axis (within-window shape preserved),
                    averaged over `--repeats` permutations

Zero-ablation is also computed but reported as a diagnostic only: setting a
standardised channel to 0 pushes the input off the training manifold, so the
resulting error increase measures distribution shift as much as importance.
Permutation is the metric to quote.

Stock back-mapping: because PCA is a linear map, the attributed weight of
stock j is sum_c |loading[j, c]| * importance[c], normalised to 100%. This is
an upper bound on interpretability -- it inherits every instability of the
importance estimates.

Usage
-----
    python notebooks/analyze_pc_attribution.py --run-dir artifacts/Run_<ts>
    python notebooks/analyze_pc_attribution.py --run-dir artifacts/Run_<ts> --repeats 50
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tensorflow import keras  # noqa: E402
from src.forecasting.lstm.data import make_cross_boundary_windowed_data  # noqa: E402

TARGET_COL = "VNINDEX"
HIST_CHANNEL = "VNINDEX_history"
# A representation is analysed only if every tuning config trained normally.
MAX_COLLAPSE_RATE = 0.0
COLLAPSE_EPOCH_THRESHOLD = 5


def _rmse(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - p) ** 2)))


def _hdr(text: str, width: int = 74) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _eligible_labels(run_dir: Path) -> list[str]:
    """Representations whose LSTM tuning grid is free of premature stops."""
    eligible, refused = [], []
    for child in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        sweep = child / "outputs" / "lstm_vnindex_sweep" / "sweep_summary.csv"
        if not sweep.exists():
            continue
        df = pd.read_csv(sweep)
        rate = float((df["Best_Epoch"] <= COLLAPSE_EPOCH_THRESHOLD).mean())
        if rate > MAX_COLLAPSE_RATE:
            refused.append((child.name, rate))
        else:
            eligible.append(child.name)
    if refused:
        print("  Refused (corrupted tuning grid — not analysable):")
        for name, rate in refused:
            print(f"    {name}: {rate*100:.0f}% of configs collapsed to an untrained model")
    return eligible


def _load_deployed_model(run_dir: Path, label: str):
    """Reconstruct the final-refit model and its exact Test tensor."""
    lsd = run_dir / label / "outputs" / "lstm_vnindex_sweep"
    bundles = sorted(lsd.glob("lstm_vnindex_lb*_bs*.pkl"))
    if not bundles:
        raise FileNotFoundError(f"No LSTM bundle under {lsd}")
    with open(bundles[0], "rb") as f:
        b = pickle.load(f)
    if b.get("model_weights") is None or b.get("model_config_json") is None:
        raise ValueError(f"{bundles[0]} carries no weights; cannot attribute.")

    model = keras.models.model_from_json(b["model_config_json"])
    model.set_weights(b["model_weights"])

    lookback = int(b["lookback"])
    pc_cols = list(b["feature_columns"])
    x_scaler, y_scaler = b["x_scaler"], b["y_scaler"]

    dp = run_dir / label / "data" / "processed"
    target = pd.read_csv(dp / "core" / "vnindex_target.csv",
                         parse_dates=["Ngày"]).set_index("Ngày")

    def scale(df: pd.DataFrame) -> pd.DataFrame:
        s = pd.DataFrame(x_scaler.transform(df[pc_cols].astype(float)),
                         index=df.index, columns=pc_cols)
        s[TARGET_COL] = y_scaler.transform(
            target.loc[df.index, [TARGET_COL]].astype(float)).ravel()
        return s

    scaled = {}
    for split in ("train", "val", "test"):
        d = pd.read_csv(dp / "pca" / f"{split}_pca.csv",
                        parse_dates=["Ngày"]).set_index("Ngày")
        scaled[split] = scale(d.join(target))
    dev = pd.concat([scaled["train"], scaled["val"]]).sort_index()

    X_pc, y_sc, X_hist, dates = make_cross_boundary_windowed_data(
        dev, scaled["test"], pc_cols, TARGET_COL, lookback)
    X = np.concatenate([X_pc, X_hist], axis=2)
    truth = y_scaler.inverse_transform(y_sc).ravel()

    # Reproduction guard: the reconstructed model must match the persisted
    # predictions, otherwise the attribution describes a different model.
    pred = y_scaler.inverse_transform(model.predict(X, verbose=0)).ravel()
    saved_path = lsd / f"predictions_lookback_{lookback}_batch_{int(b['batch_size'])}.csv"
    if saved_path.exists():
        saved = pd.read_csv(saved_path, parse_dates=["Date"]).set_index("Date")
        drift = float(np.abs(pred - saved["Predicted_VNINDEX"].to_numpy()).max())
        if drift > 1e-2:
            raise AssertionError(
                f"{label}: reconstructed predictions deviate from the persisted "
                f"ones by {drift:.4g}; refusing to attribute a model that is not "
                f"the one that produced the reported results.")
    else:
        drift = float("nan")

    return {"model": model, "X": X, "truth": truth, "dates": dates,
            "pc_cols": pc_cols, "lookback": lookback, "y_scaler": y_scaler,
            "base_rmse": _rmse(truth, pred), "repro_drift": drift,
            "loadings": pd.read_csv(dp / "pca" / "pca_loadings.csv", index_col=0)}


def channel_importance(ctx: dict, repeats: int, seed: int = 42) -> pd.DataFrame:
    model, X, truth = ctx["model"], ctx["X"], ctx["truth"]
    y_scaler, base = ctx["y_scaler"], ctx["base_rmse"]
    channels = ctx["pc_cols"] + [HIST_CHANNEL]
    rng = np.random.default_rng(seed)

    rows = []
    for c, name in enumerate(channels):
        deltas = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, :, c] = Xp[rng.permutation(len(Xp)), :, c]
            deltas.append(_rmse(truth, y_scaler.inverse_transform(
                model.predict(Xp, verbose=0)).ravel()) - base)
        Xa = X.copy()
        Xa[:, :, c] = 0.0
        abl = _rmse(truth, y_scaler.inverse_transform(
            model.predict(Xa, verbose=0)).ravel()) - base
        mean, sd = float(np.mean(deltas)), float(np.std(deltas, ddof=1))
        rows.append({
            "channel": name,
            "is_pc": name != HIST_CHANNEL,
            "perm_dRMSE_mean": mean,
            "perm_dRMSE_sd": sd,
            "signal_over_2sd": bool(mean > 2 * sd),
            "harmful": bool(mean < 0),
            "ablation_dRMSE_diagnostic": abl,
        })
    return pd.DataFrame(rows)


def attribute_to_stocks(imp: pd.DataFrame, loadings: pd.DataFrame,
                        pc_cols: list[str]) -> pd.Series:
    w = imp.set_index("channel").loc[pc_cols, "perm_dRMSE_mean"].clip(lower=0).to_numpy()
    if w.sum() <= 0:
        return pd.Series(dtype=float)
    contrib = (loadings[pc_cols].abs() * w).sum(axis=1)
    return (100 * contrib / contrib.sum()).sort_values(ascending=False)


def fig_attribution(per_label: dict, out_dir: Path) -> None:
    labels = list(per_label)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    width = 0.8 / max(1, len(labels))
    for i, lab in enumerate(labels):
        imp = per_label[lab]["importance"]
        pcs = imp[imp["is_pc"]]
        x = np.arange(len(pcs)) + i * width
        ax.bar(x, pcs["perm_dRMSE_mean"], width,
               yerr=pcs["perm_dRMSE_sd"], capsize=2,
               label=lab.replace("pca_cev_", "CEV "))
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("Principal component index")
    ax.set_ylabel("Delta Test RMSE when permuted")
    ax.set_title("(a) PC importance is unrelated to variance rank")
    ax.set_xticks(np.arange(11))
    ax.set_xticklabels([f"PC{i+1}" for i in range(11)], fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)

    ax = axes[1]
    names = [l.replace("pca_cev_", "CEV ") for l in labels]
    hist = [per_label[l]["hist_importance"] for l in labels]
    pcs = [per_label[l]["pc_importance"] for l in labels]
    x = np.arange(len(labels))
    ax.bar(x - .2, hist, .4, label="Target's own history", color="#1f4e79")
    ax.bar(x + .2, pcs, .4, label="All PCs combined", color="#2e8b57")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("Delta Test RMSE")
    ax.set_title("(b) Lagged VN-Index versus the whole PC block")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)
    for xi, (h, p) in enumerate(zip(hist, pcs)):
        if p > 0:
            ax.text(xi, max(h, p) * 1.03, f"{h/p:.1f}x", ha="center", fontsize=9)

    ax = axes[2]
    for lab in labels:
        s = per_label[lab]["stocks"]
        if s.empty:
            continue
        cum = s.cumsum().to_numpy()
        ax.plot(range(1, len(cum) + 1), cum, lw=1.8,
                label=lab.replace("pca_cev_", "CEV "))
    n = 318
    ax.plot(range(1, n + 1), np.arange(1, n + 1) * 100 / n, "k:", lw=1.2,
            label="Uniform (no concentration)")
    ax.set_xlabel("Number of stocks (ranked)")
    ax.set_ylabel("Cumulative attributed importance (%)")
    ax.set_title("(c) Stock attribution is close to uniform")
    ax.set_xlim(0, n)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    fig.suptitle("Figure A1. Exploratory LSTM attribution — single seed (42), "
                 "valid models only", y=1.03)
    fig.tight_layout()
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "figA1_pc_attribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  [FIGURE] figA1_pc_attribution.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--repeats", type=int, default=20,
                    help="Permutation repeats per channel (default 20)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    manifest = json.loads((run_dir / "sweep_manifest.json").read_text())
    out_dir = (Path(args.out_dir) if args.out_dir
               else PROJECT_ROOT / "notebooks" / "analysis_output" / manifest["run_id"])
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    _hdr(f"EXPLORATORY PC ATTRIBUTION  |  {manifest['run_id']}")
    print("  NOTE: exploratory only. Single seed (42). Not part of RQ1-RQ3 evidence.")
    print()
    labels = _eligible_labels(run_dir)
    pca_labels = [l for l in labels if l.startswith("pca_cev_")]
    print(f"\n  Analysing: {', '.join(pca_labels) if pca_labels else '(none)'}")
    if not pca_labels:
        raise SystemExit("No representation has a clean LSTM tuning grid; nothing to attribute.")

    per_label, imp_frames, stock_frames = {}, [], []
    for label in pca_labels:
        ctx = _load_deployed_model(run_dir, label)
        _hdr(f"{label}  (k={len(ctx['pc_cols'])}, lookback={ctx['lookback']}, "
             f"Test RMSE={ctx['base_rmse']:.4f})")
        print(f"  reproduction drift vs persisted predictions: {ctx['repro_drift']:.2e}")

        imp = channel_importance(ctx, repeats=args.repeats)
        for _, r in imp.iterrows():
            tag = ""
            if r["channel"] == HIST_CHANNEL:
                tag = "   <- target's own history"
            elif r["harmful"]:
                tag = "   (permuting IMPROVES the forecast)"
            elif not r["signal_over_2sd"]:
                tag = "   (not distinguishable from 0)"
            print(f"  {r['channel']:16s} dRMSE = {r['perm_dRMSE_mean']:+8.4f} "
                  f"+/- {r['perm_dRMSE_sd']:.4f}{tag}")

        hist_imp = max(0.0, float(imp.loc[imp["channel"] == HIST_CHANNEL,
                                         "perm_dRMSE_mean"].iloc[0]))
        pc_imp = float(imp.loc[imp["is_pc"], "perm_dRMSE_mean"].clip(lower=0).sum())
        stocks = attribute_to_stocks(imp, ctx["loadings"], ctx["pc_cols"])
        top20 = float(stocks.nlargest(20).sum()) if not stocks.empty else np.nan

        print(f"\n  history / all-PCs ratio      : "
              f"{hist_imp/pc_imp if pc_imp else float('nan'):.2f}x")
        print(f"  PCs with signal > 2 SD       : "
              f"{int(imp.loc[imp['is_pc'], 'signal_over_2sd'].sum())}/{len(ctx['pc_cols'])}")
        print(f"  PCs where permuting helps    : "
              f"{int(imp.loc[imp['is_pc'], 'harmful'].sum())}/{len(ctx['pc_cols'])}")
        print(f"  top-20 stock attribution     : {top20:.1f}%  "
              f"(uniform = {2000/318:.1f}%)")

        imp2 = imp.copy()
        imp2.insert(0, "label", label)
        imp_frames.append(imp2)
        if not stocks.empty:
            sf = stocks.rename("attributed_pct").to_frame()
            sf.insert(0, "label", label)
            sf.index.name = "Symbol"
            stock_frames.append(sf.reset_index())
        per_label[label] = {"importance": imp, "hist_importance": hist_imp,
                            "pc_importance": pc_imp, "stocks": stocks}

    imp_all = pd.concat(imp_frames, ignore_index=True)
    imp_all.to_csv(out_dir / "tables" / "tA1_channel_importance.csv", index=False)
    (out_dir / "tables" / "tA1_channel_importance.md").write_text(
        "**Table A1. Permutation importance of LSTM input channels "
        "(exploratory, single seed)**\n\n"
        + imp_all.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")
    print("\n  [TABLE] tA1_channel_importance.csv / .md")

    if stock_frames:
        stocks_all = pd.concat(stock_frames, ignore_index=True)
        stocks_all.to_csv(out_dir / "tables" / "tA2_stock_attribution.csv", index=False)
        print("  [TABLE] tA2_stock_attribution.csv")

        # Stability across representations is the decisive question: if the
        # ranking is not reproducible, stock-level attribution is unusable.
        wide = stocks_all.pivot(index="Symbol", columns="label", values="attributed_pct")
        rho = wide.rank().corr(method="pearson")
        rho.to_csv(out_dir / "tables" / "tA3_attribution_stability.csv")
        (out_dir / "tables" / "tA3_attribution_stability.md").write_text(
            "**Table A3. Spearman rank correlation of stock attribution "
            "across representations**\n\n"
            + rho.to_markdown(floatfmt=".3f") + "\n", encoding="utf-8")
        print("  [TABLE] tA3_attribution_stability.csv / .md")
        _hdr("STOCK-ATTRIBUTION STABILITY (Spearman rank correlation)")
        print(rho.round(3).to_string())
        offdiag = rho.to_numpy()[~np.eye(len(rho), dtype=bool)]
        print(f"\n  mean off-diagonal rho = {offdiag.mean():+.3f}")
        if offdiag.mean() < 0.5:
            print("  VERDICT: rankings do NOT agree across representations.")
            print("           Stock-level attribution is not reproducible and must")
            print("           not be reported as an economic finding.")

    fig_attribution(per_label, out_dir)
    _hdr("DONE")
    print(f"  Tables : {out_dir / 'tables'}")
    print(f"  Figures: {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
