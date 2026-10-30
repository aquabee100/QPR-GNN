import os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch

from qprgnn.parse import iter_samples
from qprgnn.runs import Topology, predict

torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))
OUTDIR = "results/regime_shift"
OURS = "QPR-GNN (ours)"
MODELS = {  # display name -> (kind, key)
    "M/M/1 (per-flow packet size)": ("analytic", "an_mm1"),
    "M/M/1/K, aggregate (= our prior alone)": ("analytic", "an_mm1k"),
    "Gradient-boosted trees": ("baseline", "gbdt"), "MLP": ("baseline", "mlp"),
    "GCN": ("baseline", "gcn"), "GAT": ("baseline", "gat"), "Graph transformer (GPS-style)": ("baseline", "gtrans"),
    "RouteNet-style (25k)": ("qprgnn", "routenet"), "RouteNet-style (87k)": ("qprgnn", "routenet_h60"),
    "RouteNet-Fermi": ("fermi", "fermi_official"),
    "MLP + prior": ("baseline", "mlp_p"), "GBDT + prior": ("baseline", "gbdt_p"), "GAT + prior": ("baseline", "gat_p"),
    "RouteNet-style + prior features": ("qprgnn", "routenet_txres_pf"),
    "Ours, prior as readout base only": ("qprgnn", "routenet_qosres_nosched"),
    OURS: ("qprgnn", "routenet_qosres_nosched_pf"),
}


def load_sim(topo_dir):
    name = os.path.basename(topo_dir.rstrip("/"))
    if not os.path.exists(f"cache/shift_{name}.pkl"):
        S = list(iter_samples(os.path.join(topo_dir, "results", name)))
        S.sort(key=lambda s: int(s["graph_file"].split("_")[1].split(".")[0]))
        pickle.dump(S, open(f"cache/shift_{name}.pkl", "wb"), protocol=4)
    return name, Topology(f"shift_{name}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    report, rows_all, tests = [], [], []
    rng = np.random.default_rng(0)
    for td in sys.argv[1:] or ["sim_inputs/Nsfnet"]:
        name, topo = load_sim(td); S = topo.S
        y, F = topo.y, topo.flows; off = np.concatenate([[0], np.cumsum(F)[:-1]])
        plan = pd.read_csv(os.path.join(td, "samples.csv")).set_index("sample")
        sid = np.array([int(s["graph_file"].split("_")[1].split(".")[0]) for s in S])
        stratum = plan.loc[sid, "stratum"].values
        scen, rows, preds = {}, [], {}
        for mname, (kind, key) in MODELS.items():
            per = []
            for tag, p in predict(kind, key, topo, "results/loto", "test_*"):
                ape = np.abs(p - y) / y
                sc = np.add.reduceat(ape, off) / F
                per.append(sc); preds[f"{mname}|{tag}"] = p.astype(np.float32)
                rows.append(dict(topology=name, model=mname, ckpt=tag, mape=100 * ape.mean(),
                                 medape=100 * np.median(ape), p90=100 * np.quantile(ape, .9),
                                 **{f"mape_{st}": 100 * np.sum((sc * F)[stratum == st]) / F[stratum == st].sum()
                                    for st in sorted(set(stratum))}))
            if per:
                scen[mname] = np.mean(per, 0)
        np.savez_compressed(f"{OUTDIR}/preds_{name}.npz", y=y, flows_per_scenario=F, sample=sid, stratum=stratum,
                            **{k.replace("/", "_"): v for k, v in preds.items()})
        df = pd.DataFrame(rows); df.to_csv(f"{OUTDIR}/{name}_metrics.csv", index=False); rows_all.append(df)
        strata = sorted(set(stratum))
        report.append(f"\n=== {name}: {len(S)} scenarios, {len(y)} flows (strata: " +
                      ", ".join(f"{st} {np.sum(stratum == st)}" for st in strata) + ") ===")
        report.append(f"{'model':42s} {'ckpts':>5s} {'MAPE':>14s} {'MedAPE':>7s} {'P90':>7s}  " + " ".join(f"{st[:14]:>14s}" for st in strata))
        for mname in MODELS:
            g = df[df.model == mname]
            if len(g):
                report.append(f"{mname:42s} {len(g):5d} {g.mape.mean():7.2f} ± {g.mape.std() if len(g) > 1 else 0:4.2f} "
                              f"{g.medape.mean():7.2f} {g.p90.mean():7.2f}  " + " ".join(f"{g[f'mape_{st}'].mean():14.2f}" for st in strata))
        for mname in MODELS:
            if mname == OURS or mname not in scen:
                continue
            d = scen[OURS] - scen[mname]; num = d * F
            est = num.sum() / F.sum()
            null = (rng.choice([-1.0, 1.0], (20000, len(d))) * num).sum(1) / F.sum()
            p = (1 + np.sum(np.abs(null) >= abs(est))) / 20001
            tests.append(dict(topology=name, model=mname, diff=100 * est, p=p, better=float((d < 0).mean())))
    T = pd.DataFrame(tests)
    pv = T.p.values; o = np.argsort(pv); holm = np.empty(len(pv)); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (len(pv) - k) * pv[i])); holm[i] = run
    T["p_holm"] = holm
    report.append("\n=== OURS vs each model (difference in flow-weighted MAPE, checkpoint-averaged single-model errors; "
                  f"sign-flip test over scenarios; Holm over {len(T)} tests) ===")
    for _, t in T.iterrows():
        report.append(f"{t.topology:16s} {t.model:42s} {t['diff']:+7.2f}  p_holm {t.p_holm:8.1e}  ours better in {100*t.better:3.0f}% of scenarios")
    w = ((T["diff"] < 0) & (T.p_holm < .05)).sum(); l = ((T["diff"] > 0) & (T.p_holm < .05)).sum()
    report.append(f"\nsignificant wins {w} / {len(T)}, significant losses {l}")
    out = "results/analysis/regime_shift.txt"
    open(out, "w").write("FROZEN models on new simulator topologies (never used for any decision)\n" + "\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
