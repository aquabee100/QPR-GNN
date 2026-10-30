import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from scipy.stats import wilcoxon

from qprgnn.runs import Topology, predict, seed_of

torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))
D = sys.argv[1] if len(sys.argv) > 1 else "results/simfold"
NS3 = "ns3fold" in D
STRATA = "cache/ns3_strata.json" if NS3 else "cache/sim_strata.json"
SUMMARY = "results/analysis/ns3_folds.txt" if NS3 else "results/analysis/sim_folds.txt"
OURS = "QPR-GNN (ours)"
MODELS = {  # display name -> (kind, key, group)
    "M/M/1/K, aggregate (= our prior alone)": ("analytic", "an_mm1k", "classical analytic"),
    "Gradient-boosted trees": ("baseline", "gbdt", "classical ML"),
    "RouteNet-style (87k)": ("qprgnn", "routenet_h60", "current GNN"),
    "RouteNet-Fermi": ("fermi", "fermi_official", "latest GNN"),
    "QT-RouteNet": ("qt", "qtroutenet", "queueing-informed GNN"),
    "QT-RouteNet + transmission term": ("qt", "qtroutenet_tx", "queueing-informed GNN"),
    "GAT": ("baseline", "gat", "modern GNN"),
    "GBDT + prior": ("baseline", "gbdt_p", "matched information"),
    "RouteNet-style + prior features": ("qprgnn", "routenet_txres_pf", "matched information"),
    "Ours, prior as readout base only": ("qprgnn", "routenet_qosres_nosched", "ablation"),
    OURS: ("qprgnn", "routenet_qosres_nosched_pf", "ours"),
}


def holm(p):
    o = np.argsort(p); h = np.empty(len(p)); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (len(p) - k) * p[i])); h[i] = run
    return h


def main():
    folds = json.load(open(f"{D}/folds.json"))
    strata = json.load(open(STRATA)) if os.path.exists(STRATA) else {}
    rows, scen_all = [], {}
    for fold, topos in folds.items():
        for T in topos.split(","):
            if not os.path.exists(f"cache/{T}.pkl"):
                print("missing cache", T); continue
            topo = Topology(T)
            y, F = topo.y, topo.flows; off = np.concatenate([[0], np.cumsum(F)[:-1]])
            st = np.array([strata.get(f"{T}|{s['source_file']}|{s['source_line']}", {}).get("stratum", "n/a") for s in topo.S])
            P = {}
            for mname, (kind, key, grp) in MODELS.items():
                per = []
                for stem, p in predict(kind, key, topo, D, fold):
                    seed = seed_of(stem) if kind != "analytic" else "analytic"; ape = np.abs(p - y) / y; sc = np.add.reduceat(ape, off) / F
                    per.append(sc); P[f"{mname}|{seed}"] = p.astype(np.float32)
                    r = dict(fold=fold, topology=T, model=mname, group=grp, seed=seed, flows=len(y),
                             mape=100 * ape.mean(), medape=100 * np.median(ape), p90=100 * np.quantile(ape, .9))
                    for s_ in sorted(set(st)):
                        r[f"mape_{s_}"] = 100 * np.sum((sc * F)[st == s_]) / F[st == s_].sum()
                    rows.append(r)
                if per:
                    scen_all[(T, mname)] = (np.mean(per, 0), F, st)
            np.savez_compressed(f"{D}/preds_{fold}_{T}.npz", y=y, flows=F, stratum=st,
                                **{k.replace("/", "_").replace(" ", "_"): v for k, v in P.items()})
            print("done", fold, T, flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{D}/metrics.csv", index=False)
    rng = np.random.default_rng(0); tests = []
    for (T, m), (sc, F, st) in scen_all.items():
        if m == OURS or (T, OURS) not in scen_all:
            continue
        d = scen_all[(T, OURS)][0] - sc; num = d * F; est = num.sum() / F.sum()
        null = (rng.choice([-1.0, 1.0], (20000, len(d))) * num).sum(1) / F.sum()
        tests.append(dict(topology=T, model=m, diff=100 * est, p=(1 + np.sum(np.abs(null) >= abs(est))) / 20001,
                          better=float((d < 0).mean())))
    Tt = pd.DataFrame(tests); Tt["p_holm"] = holm(Tt.p.values); Tt.to_csv(f"{D}/tests.csv", index=False)
    seeds_mean = df.groupby(["topology", "model"]).agg(mape=("mape", "mean"), flows=("flows", "first"))
    out = [("ns-3" if NS3 else "BNNetSimulator") + " cross-topology folds: FROZEN METHOD retrained on 12 topologies, tested on 4 held-out (4 folds).",
           "MAPE % per held-out topology (mean over seeds); avg = mean over topologies; pooled = flow-weighted.\n"]
    topos = sorted(set(df.topology), key=lambda t: list(df.topology).index(t))
    out.append(f"{'model':40s}" + "".join(f"{t[4:14]:>11s}" for t in topos) + f"{'avg':>8s}{'pooled':>8s}")
    for m in MODELS:
        v = [seeds_mean.loc[(t, m), "mape"] if (t, m) in seeds_mean.index else np.nan for t in topos]
        w = [seeds_mean.loc[(t, m), "flows"] if (t, m) in seeds_mean.index else 0 for t in topos]
        out.append(f"{m:40s}" + "".join(f"{x:11.2f}" for x in v) + f"{np.nanmean(v):8.2f}{np.nansum(np.array(v) * w) / max(sum(w), 1):8.2f}")
    sc_cols = [c for c in df.columns if c.startswith("mape_")]
    out.append("\nMAPE % by declared load stratum (pooled over held-out topologies, mean over seeds)")
    g = df.groupby("model")[sc_cols].mean()
    out.append(f"{'model':40s}" + "".join(f"{c[5:20]:>17s}" for c in sc_cols))
    for m in MODELS:
        if m in g.index:
            out.append(f"{m:40s}" + "".join(f"{g.loc[m, c]:17.2f}" for c in sc_cols))
    out.append(f"\nOURS vs each model per held-out topology (sign-flip test over scenarios, Holm over {len(Tt)} tests)")
    summ = Tt.assign(win=(Tt["diff"] < 0) & (Tt.p_holm < .05), loss=(Tt["diff"] > 0) & (Tt.p_holm < .05)) \
             .groupby("model").agg(mean_diff=("diff", "mean"), wins=("win", "sum"), losses=("loss", "sum"), n=("diff", "size"))
    for m, r in summ.iterrows():
        out.append(f"{m:40s} mean diff {r.mean_diff:+7.2f}   significant wins {int(r.wins)}/{int(r.n)}, losses {int(r.losses)}")
    out.append(f"\nTOTAL: significant wins {int(summ.wins.sum())} / {len(Tt)}, significant losses {int(summ.losses.sum())}")
    out.append("\nOURS vs each model across the held-out topologies (per-topology MAPE, mean over seeds; Wilcoxon signed-rank)")
    ours = np.array([seeds_mean.loc[(t, OURS), "mape"] for t in topos])
    for m in MODELS:
        if m == OURS:
            continue
        v = np.array([seeds_mean.loc[(t, m), "mape"] for t in topos])
        out.append(f"{m:40s} mean diff {np.mean(ours - v):+7.2f} ({100 * (np.mean(ours) / np.mean(v) - 1):+6.1f}%)   "
                   f"ours better on {int((ours < v).sum())}/{len(topos)}   p = {wilcoxon(ours, v).pvalue:.4f}")
    os.makedirs("results/analysis", exist_ok=True)
    open(SUMMARY, "w").write("\n".join(out) + "\n"); print("\n".join(out))


if __name__ == "__main__":
    main()
