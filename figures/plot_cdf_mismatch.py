from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "results"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a6fd1"]
GRAY, INK, MUTED, GRID = "#a9a8a0", "#1f1f1e", "#6b6a63", "#e4e3dd"
OURS, OURS_C = "QPR-GNN (ours)", SERIES[2]
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "pdf.fonttype": 42,
                     "font.family": "DejaVu Sans"})
COL = {"RouteNet-style (87k)": "#6b6a63", "RouteNet-Fermi": "#eda100", "MSMP + attention": "#2a78d6", OURS: OURS_C}
cdf = pd.read_csv(ROOT / "ape_cdf_v3.csv", index_col=0)
mis = pd.read_csv(ROOT / "error_by_size_mismatch_v3.csv", index_col=0)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1.0]))
ax = axs[0]
for m, c, ls in (("M/M/1/K, aggregate (= our prior alone)", GRAY, "--"), ("RouteNet-style (87k)", COL["RouteNet-style (87k)"], "-"),
                 ("RouteNet-Fermi", COL["RouteNet-Fermi"], "-"), ("MSMP + attention", COL["MSMP + attention"], "-"), (OURS, OURS_C, "-")):
    lab = "M/M/1/K (prior alone)" if m.startswith("M/M/1/K") else m
    ax.plot(cdf[m].values, cdf.index.values / 100, color=c, ls=ls, lw=1.6 if m == OURS else 1.1, label=lab)
ax.set_xscale("log"); ax.set_xlim(0.1, 100); ax.set_ylim(0, 1)
ax.set_xlabel("absolute percentage error (%)"); ax.set_ylabel("fraction of held-out flows")
ax.legend(frameon=False, fontsize=7.2, loc="upper left")
ax.set_title("(a) Distribution of per-flow error", fontsize=9.0, color=INK, pad=4)
ax = axs[1]
labels = ["<0.1", "0.1–0.3", "0.3–0.6", "0.6–1.0"]
mis = mis.iloc[:4]
x = np.arange(len(mis.index))
for m, c in (("RouteNet-style (87k)", COL["RouteNet-style (87k)"]), ("RouteNet-Fermi", COL["RouteNet-Fermi"]), (OURS, OURS_C)):
    ax.plot(x, mis[m].values, "-o", color=c, ms=3.5, lw=1.6 if m == OURS else 1.1, label=m)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.2)
ax.set_xlabel(r"packet-size mismatch $\max_k|\log(s_f/\bar s)|$"); ax.set_ylabel("held-out MAPE (%)")
ax.legend(frameon=False, fontsize=7.2, loc="upper left")
ax.set_title("(b) Error against packet-size mismatch", fontsize=9.0, color=INK, pad=4)
fig.tight_layout(); fig.savefig(Path(__file__).resolve().parent / "fig_cdf_mismatch.pdf", metadata={"CreationDate": None}); plt.close(fig)
print("written fig_cdf_mismatch.pdf")
