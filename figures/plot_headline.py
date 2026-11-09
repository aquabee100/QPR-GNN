from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
OURS = "QPR-GNN (ours)"
BENCH = [("Archive (3)", "#2a78d6", "s"), ("BNNetSimulator (16)", "#eda100", "o"),
         ("ns-3 (16)", "#1baf7a", "^"), ("IoT traffic (16)", "#e34948", "D")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def per_topology():
    a = pd.read_csv(ROOT / "results/loto_metrics_v3.csv").groupby(["model", "fold"]).mape.mean().unstack()
    rows = [("Archive (3)", a)]
    for name, d in (("BNNetSimulator (16)", "simfold_mape"), ("ns-3 (16)", "ns3fold_mape"), ("IoT traffic (16)", "iotfold")):
        rows.append((name, pd.read_csv(ROOT / f"results/{d}/metrics.csv").groupby(["model", "topology"]).mape.mean().unstack()))
    return rows


def main():
    data = per_topology()
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    rec = []
    rng = np.random.default_rng(0)
    for ax, base, title in zip(axs, ("RouteNet-Fermi", "MSMP + attention"),
                               ("(a) vs. RouteNet-Fermi (2023)", "(b) vs. MSMP + attention (2024)")):
        wins = n = 0
        for i, ((name, t), (_, color, mk)) in enumerate(zip(data, BENCH)):
            x, y = t.loc[base].values, t.loc[OURS].values
            red = 100 * (x - y) / x
            jit = (rng.random(len(red)) - 0.5) * 0.36
            ax.scatter(i + jit, red, s=24, marker=mk, color=color, edgecolor="white", linewidth=0.6, zorder=3)
            med = np.median(red)
            ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=1.4, zorder=4)
            ax.text(i + 0.31, med, f"{med:.0f}%", fontsize=7.2, color=INK, va="center", ha="left")
            wins += int((red > 0).sum()); n += len(red)
            rec += [dict(panel=base, benchmark=name, topology=c, baseline=xv, ours=yv, reduction=r)
                    for c, xv, yv, r in zip(t.columns, x, y, red)]
        ax.axhline(0, color=MUTED, lw=0.9, ls="--", zorder=1)
        ax.set_xticks(range(len(BENCH))); ax.set_xticklabels(["Archive\n(3)", "BNNet-\nSimulator (16)", "ns-3\n(16)", "IoT traffic\n(16)"], fontsize=7.4)
        ax.set_xlim(-0.55, len(BENCH) - 0.25)
        ax.set_title(title, fontsize=9.0, color=INK, pad=4)
        ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.text(0.02, 0.97, f"QPR-GNN more accurate on {wins} of {n} held-out topologies",
                transform=ax.transAxes, fontsize=7.6, color=INK, va="top", ha="left")
        ax.text(-0.5, -1.2, "below line: baseline better", fontsize=7.2, color=MUTED, ha="left", va="top")
    axs[0].set_ylabel("error reduction by QPR-GNN (%)")
    lo = min(r["reduction"] for r in rec); hi = max(r["reduction"] for r in rec)
    axs[0].set_ylim(min(-8, lo - 2), hi + 9)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig_headline.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / "fig_headline.png", dpi=200)
    pd.DataFrame(rec).to_csv(OUT / "fig_headline.csv", index=False)
    print("points:", len(rec))


if __name__ == "__main__":
    main()
