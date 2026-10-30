import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch

from qprgnn.protocol import TARGETS
from qprgnn.runs import Topology, predict, run_files, seed_of

torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))
RUNS, PRED, OUT = "results/loto", "results/preds", "results/analysis"
OURS = "QPR-GNN (ours)"
PREV = "Ours, prior as readout base only"
# display name -> (group, kind, key)
MODELS = {
    "M/M/1 (per-flow packet size)": ("classical analytic", "analytic", "an_mm1"),
    "M/M/1/K, aggregate (= our prior alone)": ("classical analytic", "analytic", "an_mm1k"),
    "M/G/1 (Pollaczek-Khinchine)": ("classical analytic", "analytic", "an_mg1"),
    "MLP": ("classical ML", "baseline", "mlp"),
    "Gradient-boosted trees": ("classical ML", "baseline", "gbdt"),
    "RouteNet-style (25k)": ("current GNN", "qprgnn", "routenet"),
    "RouteNet-style (87k)": ("current GNN", "qprgnn", "routenet_h60"),
    "RouteNet-Fermi": ("latest GNN", "fermi", "fermi_official"),
    "QT-RouteNet": ("queueing-informed GNN", "qt", "qtroutenet"),
    "QT-RouteNet + transmission term": ("queueing-informed GNN", "qt", "qtroutenet_tx"),
    "GCN": ("modern GNN", "baseline", "gcn"),
    "GAT": ("modern GNN", "baseline", "gat"),
    "Graph transformer (GPS-style)": ("modern GNN", "baseline", "gtrans"),
    "MLP + prior": ("matched information", "baseline", "mlp_p"),
    "GBDT + prior": ("matched information", "baseline", "gbdt_p"),
    "GAT + prior": ("matched information", "baseline", "gat_p"),
    "RouteNet-style + prior features": ("matched information", "qprgnn", "routenet_txres_pf"),
    PREV: ("ablation", "qprgnn", "routenet_qosres_nosched"),
    OURS: ("ours", "qprgnn", "routenet_qosres_nosched_pf"),
}


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")[:60]


def all_preds(name, kind, key, fold, topo):
    f = f"{PRED}/{slug(name)}__{fold}.npz"
    n = 1 if kind == "analytic" else len(run_files(kind, key, RUNS, fold))
    if os.path.exists(f) and not os.environ.get("RECOMPUTE"):
        z = np.load(f)
        if len(z.files) == n:
            return {int(k[1:]): z[k] for k in z.files}
    out = {seed_of(stem): p for stem, p in predict(kind, key, topo, RUNS, fold)}
    if out:
        np.savez_compressed(f, **{f"s{k}": v for k, v in out.items()})
    return out


def scen_mape(pred, y, F):
    ape = np.abs(pred - y) / y
    return np.add.reduceat(ape, np.concatenate([[0], np.cumsum(F)[:-1]])) / F


def scores(p, y):
    ape = np.abs(p - y) / y
    return dict(mape=100 * ape.mean(), medape=100 * np.median(ape), p90=100 * np.quantile(ape, 0.9),
                logmae=float(np.abs(np.log(np.maximum(p, 1e-15)) - np.log(y)).mean()))


def holm(p):
    o = np.argsort(p); h = np.empty(len(p)); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (len(p) - k) * p[i])); h[i] = run
    return h


def main():
    os.makedirs(PRED, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    rows, scen, ens, meta = [], {}, [], {}
    for fold in TARGETS:
        topo = Topology(fold)
        y, F = topo.y, topo.flows
        arch = np.array([s["source_file"] for s in topo.S])
        np.savez(f"{PRED}/meta_{fold}.npz", y=y, tos=torch.cat([it["tos"] for it in topo.items]).numpy(),
                 flows_per_scenario=F, archive=arch, line=np.array([s["source_line"] for s in topo.S]))
        meta[fold] = (F, arch)
        for name, (grp, kind, key) in MODELS.items():
            P = all_preds(name, kind, key, fold, topo)
            if not P:
                print("missing", name, fold, flush=True); continue
            for seed, p in P.items():
                rows.append(dict(model=name, group=grp, fold=fold, seed=seed, **scores(p, y)))
            scen[(name, fold)] = np.mean([scen_mape(p, y, F) for p in P.values()], 0)
            pe = np.mean(list(P.values()), 0)
            ens.append(dict(model=name, fold=fold, ens_mape=100 * float(np.mean(np.abs(pe - y) / y))))
            print(f"{fold:15s} {name:42s} seeds {len(P)}", flush=True)
    df = pd.DataFrame(rows); df.to_csv("results/loto_metrics.csv", index=False)
    E = pd.DataFrame(ens)
    order = [m for m in MODELS if m in set(df.model)]
    W = 44
    with open(f"{OUT}/loto_table.txt", "w") as fd:
        fd.write("Leave-one-topology-out on the archive: single-model accuracy on the COMPLETE held-out topology.\n"
                 "Metric per seed, mean ± std over seeds (n). avg = mean over the 3 topologies; pooled = flow-weighted.\n")
        for met, lab in [("mape", "MAPE %"), ("medape", "median APE %"), ("p90", "90th-percentile APE %"), ("logmae", "log-MAE")]:
            g = df.groupby(["model", "fold"])[met].agg(["mean", "std", "count"])
            fd.write(f"\n=== {lab} ===\n{'model':{W}s}" + "".join(f"{f:>21s}" for f in TARGETS) + f"{'avg':>8s}\n")
            for m in order:
                cells, means = [], []
                for f in TARGETS:
                    r = g.loc[(m, f)]; means.append(r["mean"])
                    cells.append(f"{r['mean']:8.3f} ± {0 if np.isnan(r['std']) else r['std']:5.3f} ({int(r['count'])})")
                fd.write(f"{m:{W}s}" + "".join(f"{c:>21s}" for c in cells) + f"{np.mean(means):8.3f}\n")
        fd.write("\n=== pooled MAPE % (all held-out flows, flow-weighted; mean over seeds) ===\n")
        nflow = {f: meta[f][0].sum() for f in TARGETS}
        for m in order:
            g = df[df.model == m].groupby("fold")["mape"].mean()
            fd.write(f"{m:{W}s}{sum(g[f] * nflow[f] for f in TARGETS) / sum(nflow.values()):8.3f}\n")
        fd.write("\n=== ENSEMBLE of the seeds (predictions averaged; a different estimand, for reference only) ===\n")
        for m in order:
            e = E[E.model == m].set_index("fold")
            fd.write(f"{m:{W}s}" + "".join(f"{e.loc[f, 'ens_mape']:21.3f}" for f in TARGETS) + "\n")
    print(open(f"{OUT}/loto_table.txt").read())

    tests, rng = [], np.random.default_rng(0)
    for m in order:
        if m == OURS:
            continue
        for f in TARGETS:
            F, arch = meta[f]
            d = scen[(OURS, f)] - scen[(m, f)]                      # per-scenario, seed-averaged single-model errors
            ua, inv = np.unique(arch, return_inverse=True)
            num = np.bincount(inv, weights=d * F); den = np.bincount(inv, weights=F)
            est = num.sum() / den.sum()                             # = difference of the two table MAPEs
            idx = rng.integers(0, len(ua), (4000, len(ua)))
            bs = num[idx].sum(1) / den[idx].sum(1)
            null = (rng.choice([-1.0, 1.0], (20000, len(ua))) * num).sum(1) / den.sum()
            tests.append(dict(model=m, fold=f, ours=100 * float((scen[(OURS, f)] * F).sum() / F.sum()),
                              other=100 * float((scen[(m, f)] * F).sum() / F.sum()), diff=100 * est,
                              lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5),
                              p=(1 + np.sum(np.abs(null) >= abs(est))) / (1 + len(null)),
                              better=float((num / den < 0).mean()), units=len(ua)))
    T = pd.DataFrame(tests); T["p_holm"] = holm(T.p.values)
    T.to_csv("results/loto_tests.csv", index=False)
    with open(f"{OUT}/loto_significance.txt", "w") as fd:
        fd.write("OURS vs each model. Estimand: single-model flow-weighted MAPE difference on the complete held-out set\n"
                 "(per-seed errors averaged over seeds; equals the difference of the table entries). Unit = source archive.\n"
                 "95% CI: archive cluster bootstrap. p: paired sign-flip randomisation test of the same statistic\n"
                 f"(20,000 draws). Holm over {len(T)} tests. 'ours better' = share of archives where ours is better.\n\n")
        fd.write(f"{'model':{W}s} {'topology':15s} {'units':>5s} {'ours':>7s} {'model':>7s} {'diff':>7s} {'95% CI':>17s} {'p_holm':>9s} {'ours better':>11s}\n")
        for _, t in T.iterrows():
            fd.write(f"{t.model:{W}s} {t.fold:15s} {t.units:5d} {t.ours:7.3f} {t.other:7.3f} {t['diff']:+7.3f} "
                     f"[{t.lo:+6.2f},{t.hi:+6.2f}] {t.p_holm:9.1e} {100 * t.better:10.0f}%\n")
        win = T[(T["diff"] < 0) & (T.p_holm < 0.05)]; loss = T[(T["diff"] > 0) & (T.p_holm < 0.05)]
        fd.write(f"\nsignificant wins for OURS: {len(win)} / {len(T)}; significant losses: {len(loss)}; "
                 f"not significant: {len(T) - len(win) - len(loss)}\n")
        for _, t in T[(T.p_holm >= 0.05)].iterrows():
            fd.write(f"  not significant: {t.model} on {t.fold} ({t['diff']:+.3f})\n")
        for _, t in loss.iterrows():
            fd.write(f"  LOSS: {t.model} on {t.fold} ({t['diff']:+.3f})\n")
    print(open(f"{OUT}/loto_significance.txt").read())


if __name__ == "__main__":
    main()
