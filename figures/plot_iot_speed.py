from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
OURS = "QPR-GNN (ours)"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
MODELS = [("RouteNet-Fermi", "RouteNet-Fermi", "#eda100"), ("MSMP + attention", "MSMP + attention", "#2a78d6"),
          ("QT-RouteNet + transmission term", "QT-RouteNet + tx", "#e34948"), (OURS, "QPR-GNN", "#1baf7a")]
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def main():
    rec = []
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.45, 1.0]))
    C = pd.read_csv(ROOT / "results/iotfold/metrics_by_tos.csv").groupby(["model", "tos"])[["ape_sum", "n"]].sum()
    classes = [(0, "control / alarm"), (1, "telemetry"), (2, "video")]
    ax = axs[0]; w = 0.2
    for j, (m, lab, col) in enumerate(MODELS):
        v = [100 * C.loc[(m, c), "ape_sum"] / C.loc[(m, c), "n"] for c, _ in classes]
        x = np.arange(len(classes)) + (j - 1.5) * (w + 0.045)
        ax.bar(x, v, width=w, color=col, label=lab, zorder=3)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.3, f"{vi:.1f}", ha="center", va="bottom", fontsize=7.2, color=INK,
                    fontweight="bold" if m == OURS else "normal")
        rec += [dict(panel="iot_class", model=m, x=cl, y=vi) for (_, cl), vi in zip(classes, v)]
    ax.set_xticks(range(len(classes))); ax.set_xticklabels([c for _, c in classes]); ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel("MAPE on IoT traffic (%)"); ax.set_ylim(0, 31)
    ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("(a) Error by IoT traffic class", fontsize=9.0, color=INK, pad=4)
    ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="upper center", handlelength=1.0, columnspacing=1.2)
    ax = axs[1]
    E = pd.read_csv(ROOT / "results/efficiency_v3.csv")
    E = E[E.topology == "ns3_Palmetto"].set_index("model")
    A = pd.read_csv(ROOT / "results/ns3fold_mape/metrics.csv").groupby(["model", "topology"]).mape.mean().groupby("model").mean()
    OFFSET = {"RouteNet-Fermi": (0.88, 0.0, "right", "center"), "QT-RouteNet + transmission term": (1.12, -0.12, "left", "top"),
              "MSMP + attention": (0.88, 0.12, "right", "bottom"), OURS: (0.88, 0.0, "right", "center")}
    pts = [("RouteNet-style (87k)", "RouteNet-style (87k)", "#a9a8a0"), ("GAT", "GAT", "#a9a8a0")] + MODELS
    for m, lab, col in pts:
        if m not in E.index or m not in A.index:
            continue
        x, y = E.loc[m, "median_ms"], A.loc[m]
        ax.scatter(x, y, s=42 if m == OURS else 28, color=col, edgecolor="white", linewidth=0.7, zorder=3)
        dx, dy, ha, va = OFFSET.get(m, (1.12, 0.0, "left", "center"))
        ax.text(x * dx, y + dy, lab, fontsize=7.2, color=INK, va=va, ha=ha, fontweight="bold" if m == OURS else "normal")
        rec.append(dict(panel="speed", model=m, x=x, y=y))
    ax.set_xscale("log"); ax.set_xlim(6, 1200)
    ax.set_xlabel("CPU time per network state, Palmetto (ms)"); ax.set_ylabel("ns-3 MAPE (%)")
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("(b) Accuracy against inference cost", fontsize=9.0, color=INK, pad=4)
    ax.set_ylim(8.0, 13.8)
    ax.text(0.97, 0.96, "better: lower left", transform=ax.transAxes, fontsize=7.2, color=MUTED, ha="right", va="top")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig_iot_speed.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / "fig_iot_speed.png", dpi=200)
    pd.DataFrame(rec).to_csv(OUT / "fig_iot_speed.csv", index=False)
    print("written", len(rec))


if __name__ == "__main__":
    main()
