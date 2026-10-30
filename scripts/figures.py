import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a6fd1"]
GRAY, INK, MUTED, GRID = "#a9a8a0", "#1f1f1e", "#6b6a63", "#e4e3dd"
OURS_C = SERIES[2]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "pdf.fonttype": 42})
FOLDS = [("test_abilene", "Abilene"), ("test_nobel", "Nobel-EU"), ("test_germany50", "Germany50")]
OURS = "QPR-GNN (ours)"
GROUP_C = {"classical analytic": GRAY, "classical ML": SERIES[4], "current GNN": SERIES[0], "latest GNN": SERIES[3],
           "queueing-informed GNN": "#b07b4f", "modern GNN": SERIES[1], "matched information": SERIES[5],
           "ablation": "#7fcfb0", "ours": OURS_C}
os.makedirs("figures", exist_ok=True)


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"figures/{name}.pdf", metadata={"CreationDate": None}); fig.savefig(f"figures/{name}.png", dpi=220); plt.close(fig)


df = pd.read_csv("results/loto_metrics.csv")
grp = df.drop_duplicates("model").set_index("model")["group"]

# 1. per-topology comparison
show = ["M/M/1/K, aggregate (= our prior alone)", "RouteNet-style (87k)", "RouteNet-Fermi",
        "QT-RouteNet + transmission term", "GBDT + prior", "RouteNet-style + prior features", OURS]
cols = [GRAY, SERIES[0], SERIES[3], "#b07b4f", SERIES[5], "#b9a6e8", OURS_C]
g = df.groupby(["model", "fold"])["mape"].agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(7.0, 2.6)); w = 0.8 / len(show)
for i, (m, c) in enumerate(zip(show, cols)):
    mu = np.array([g.loc[(m, f), "mean"] for f, _ in FOLDS]); sd = np.nan_to_num([g.loc[(m, f), "std"] for f, _ in FOLDS])
    x = np.arange(3) + (i - (len(show) - 1) / 2) * w
    ax.bar(x, mu, w * 0.9, color=c, label=m, yerr=sd, error_kw=dict(lw=0.7, ecolor=MUTED, capsize=1.2))
    if m == OURS:
        for xi, v in zip(x, mu):
            ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6.5, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in FOLDS])
ax.set_ylabel("MAPE (%) on held-out topology")
ax.legend(frameon=False, ncol=4, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
save(fig, "fig_loto_main")

# 2. all models, three-topology average
per_seed = df.groupby(["model", "seed"])["mape"].mean()
order = sorted(per_seed.index.get_level_values(0).unique(), key=lambda n: -per_seed[n].mean())
XMAX = 16.5                                                       # QT-RouteNet (33) is drawn clipped
fig, ax = plt.subplots(figsize=(5.6, 4.3))
for y, n in enumerate(order):
    v = per_seed[n]; mu = v.mean(); clip = mu > XMAX
    ax.barh(y, min(mu, XMAX), xerr=0 if clip or len(v) < 2 else v.std(), color=GROUP_C[grp[n]], height=0.66,
            error_kw=dict(lw=0.6, ecolor=MUTED, capsize=1.2))
    ax.annotate(f"{mu:.2f}" + (" (axis clipped)" if clip else ""), (min(mu, XMAX), y), xytext=(-60 if clip else 4, 0),
                textcoords="offset points", va="center", fontsize=6.3, color="white" if clip else INK)
ax.set_xlim(0, XMAX)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6.3)
ax.set_xlabel("MAPE (%), average of the 3 held-out topologies")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_C.values()]
ax.legend(handles, list(GROUP_C), frameon=False, fontsize=5.8, loc="upper right")
save(fig, "fig_loto_all_models")

# 3. prior ablation
ab = pd.read_csv("results/prior_ablation.csv")
names = list(dict.fromkeys(ab.prior))
vals = [ab[ab.prior == n].groupby("fold").mape.mean().mean() for n in names]
fig, ax = plt.subplots(figsize=(4.8, 2.2))
cc = [GRAY] * (len(names) - 1) + [OURS_C]
ax.barh(range(len(names))[::-1], vals, color=cc, height=0.6)
for y, v in zip(range(len(names))[::-1], vals):
    ax.annotate(f"{v:.2f}", (v, y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.8, color=INK)
ax.set_yticks(range(len(names))[::-1]); ax.set_yticklabels(names, fontsize=6.3)
ax.set_xlabel("held-out MAPE (%), mean of 3 topologies"); ax.set_xlim(0, max(vals) * 1.15)
save(fig, "fig_prior_ablation")

# 4. simulator folds
L = open("results/analysis/sim_folds.txt").read().splitlines()
k0 = next(i for i, l in enumerate(L) if l.startswith("model") and "avg" in l)
avg = {}
for l in L[k0 + 1:]:
    if not l.strip():
        break
    avg[l[:40].strip()] = float(l.split()[-2])
k1 = next(i for i, l in enumerate(L) if l.startswith("model") and "overload" in l)
strata = L[k1].split()[1:]; st = {}
for l in L[k1 + 1:]:
    if not l.strip():
        break
    st[l[:40].strip()] = [float(x) for x in l[40:].split()]
sim_show = ["M/M/1/K, aggregate (= our prior alone)", "Gradient-boosted trees", "GAT", "RouteNet-style (87k)", "QT-RouteNet",
            "RouteNet-Fermi", "QT-RouteNet + transmission term", "RouteNet-style + prior features", "GBDT + prior", OURS]
sim_show = [m for m in sim_show if m in avg]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.0, 1.25]))
ax = axs[0]
o = sorted(sim_show, key=lambda m: -avg[m])
ax.barh(range(len(o)), [avg[m] for m in o], color=[OURS_C if m == OURS else GRAY for m in o], height=0.62)
for y, m in enumerate(o):
    ax.annotate(f"{avg[m]:.2f}", (avg[m], y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.2, color=INK)
ax.set_yticks(range(len(o))); ax.set_yticklabels(o, fontsize=6.0)
ax.set_xlabel("MAPE (%), mean of 16 held-out topologies")
ax = axs[1]
pick = ["RouteNet-style (87k)", "RouteNet-Fermi", "QT-RouteNet + transmission term", "GBDT + prior", OURS]
cols2 = [SERIES[0], SERIES[3], "#b07b4f", SERIES[5], OURS_C]
w = 0.8 / len(pick)
for i, (m, c) in enumerate(zip(pick, cols2)):
    x = np.arange(len(strata)) + (i - (len(pick) - 1) / 2) * w
    ax.bar(x, st[m], w * 0.9, color=c, label=m)
ax.set_xticks(range(len(strata))); ax.set_xticklabels([s.replace("_", " ") for s in strata], fontsize=6.5)
ax.set_ylabel("MAPE (%) by declared load stratum")
ax.set_ylim(0, max(max(st[m]) for m in pick) * 1.45)
ax.legend(frameon=False, fontsize=5.6, ncol=2, loc="upper left")
save(fig, "fig_sim_folds")
print("figures written:", sorted(os.listdir("figures")))

# 5. both simulators side by side
if os.path.exists("results/analysis/ns3_folds.txt"):
    def sim_avg(path):
        L = open(path).read().splitlines()
        k = next(i for i, l in enumerate(L) if l.startswith("model") and "avg" in l); out = {}
        for l in L[k + 1:]:
            if not l.strip():
                break
            out[l[:40].strip()] = float(l.split()[-2])
        return out
    A, B = sim_avg("results/analysis/sim_folds.txt"), sim_avg("results/analysis/ns3_folds.txt")
    show = ["RouteNet-style (87k)", "QT-RouteNet", "RouteNet-Fermi", "QT-RouteNet + transmission term",
            "GBDT + prior", "RouteNet-style + prior features", OURS]
    cols = [SERIES[0], "#d9b38c", SERIES[3], "#b07b4f", SERIES[5], "#b9a6e8", OURS_C]
    fig, ax = plt.subplots(figsize=(7.0, 2.4)); w = 0.8 / len(show)
    for i, (m, c) in enumerate(zip(show, cols)):
        x = np.arange(2) + (i - (len(show) - 1) / 2) * w
        v = [A[m], B[m]]
        ax.bar(x, v, w * 0.9, color=c, label=m)
        if m == OURS:
            for xi, vi in zip(x, v):
                ax.annotate(f"{vi:.2f}", (xi, vi), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(2)); ax.set_xticklabels(["BNNetSimulator (16 topologies)", "ns-3 (same 16 topologies)"])
    ax.set_ylabel("MAPE (%), mean of 16\nheld-out topologies")
    ax.legend(frameon=False, ncol=4, fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    save(fig, "fig_two_simulators")
    print("figure written: fig_two_simulators")

# 6. error CDF and size mismatch
if os.path.exists("results/ape_cdf.csv") and os.path.exists("results/error_by_size_mismatch.csv"):
    cdf = pd.read_csv("results/ape_cdf.csv", index_col=0)
    mis = pd.read_csv("results/error_by_size_mismatch.csv", index_col=0)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1.0]))
    ax = axs[0]
    for m, c, ls in (("M/M/1/K, aggregate (= our prior alone)", GRAY, "--"), ("RouteNet-style (87k)", SERIES[0], "-"),
                     ("RouteNet-Fermi", SERIES[3], "-"), ("GBDT + prior", SERIES[5], "-"), (OURS, OURS_C, "-")):
        ax.plot(cdf[m].values, cdf.index.values / 100, color=c, ls=ls, lw=1.6 if m == OURS else 1.1, label=m)
    ax.set_xscale("log"); ax.set_xlim(0.1, 100); ax.set_ylim(0, 1)
    ax.set_xlabel("absolute percentage error (%)"); ax.set_ylabel("fraction of held-out flows")
    ax.legend(frameon=False, fontsize=5.6, loc="upper left")
    ax = axs[1]
    labels = ["<0.1", "0.1–0.3", "0.3–0.6", "0.6–1.0"]
    mis = mis.iloc[:4]  # first four bins (96% of flows)
    x = np.arange(len(mis.index))
    for m, c in (("RouteNet-style (87k)", SERIES[0]), ("RouteNet-Fermi", SERIES[3]), (OURS, OURS_C)):
        ax.plot(x, mis[m].values, "-o", color=c, ms=3.5, lw=1.6 if m == OURS else 1.1, label=m)
    ax.set_xticks(x); ax.set_xticklabels(labels[:len(x)], fontsize=6.5)
    ax.set_xlabel(r"packet-size mismatch $\max_k|\log(s_f/\bar s)|$"); ax.set_ylabel("held-out MAPE (%)")
    ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    save(fig, "fig_cdf_mismatch")
    print("figure written: fig_cdf_mismatch")

# 7. load strata on both simulators
if os.path.exists("results/analysis/ns3_folds.txt"):
    def strata(path):
        L = open(path).read().splitlines()
        k = next(i for i, l in enumerate(L) if l.startswith("model") and "overload" in l)
        names = L[k].split()[1:]; out = {}
        for l in L[k + 1:]:
            if not l.strip():
                break
            out[l[:40].strip()] = [float(v) for v in l[40:].split()]
        return names, out
    pick = ["RouteNet-style (87k)", "RouteNet-Fermi", "QT-RouteNet + transmission term", "GBDT + prior", OURS]
    cols2 = [SERIES[0], SERIES[3], "#b07b4f", SERIES[5], OURS_C]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=False)
    for ax, (path, title) in zip(axs, (("results/analysis/sim_folds.txt", "BNNetSimulator"), ("results/analysis/ns3_folds.txt", "ns-3"))):
        names, S = strata(path); w = 0.8 / len(pick)
        for i, (m, c) in enumerate(zip(pick, cols2)):
            ax.bar(np.arange(len(names)) + (i - (len(pick) - 1) / 2) * w, S[m], w * 0.9, color=c, label=m)
        ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace("_", " ") for n in names], fontsize=6.5)
        ax.set_title(title, fontsize=8); ax.set_ylabel("MAPE (%)")
    axs[0].legend(frameon=False, fontsize=5.6, loc="upper left")
    save(fig, "fig_strata")
    print("figure written: fig_strata")
