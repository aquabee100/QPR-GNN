import os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

OURS = "QPR-GNN (ours)"
PUBLISHED = [
    ("M/M/1", "M/M/1 (per-flow packet size)", "Analytic"),
    ("M/M/1/K", "M/M/1/K, aggregate (= our prior alone)", "Analytic"),
    ("M/G/1", "M/G/1 (Pollaczek-Khinchine)", "Analytic"),
    ("MLP", "MLP", "Classical ML"),
    ("Gradient-boosted trees", "Gradient-boosted trees", "Classical ML"),
    ("RouteNet-style (25k)", "RouteNet-style (25k)", "GNN"),
    ("RouteNet-style (87k)", "RouteNet-style (87k)", "GNN"),
    ("RouteNet-Fermi", "RouteNet-Fermi", "GNN"),
    ("MSMP + attention (2024)", "MSMP + attention", "GNN"),
    ("QT-RouteNet", "QT-RouteNet", "GNN"),
    ("QT-RouteNet + tx term*", "QT-RouteNet + transmission term", "GNN"),
    ("GCN", "GCN", "GNN"),
    ("GAT", "GAT", "GNN"),
    ("Graph transformer (GPS)", "Graph transformer (GPS-style)", "GNN"),
]
ABLATION = [
    ("Prior alone (M/M/1/K)", "M/M/1/K, aggregate (= our prior alone)", "Prior"),
    ("QPR-GNN without prior features", "Ours, prior as readout base only", "QPR-GNN components"),
    ("QPR-GNN with log-ratio loss", "QPR-GNN with log-ratio loss", "QPR-GNN components"),
    ("MLP + prior", "MLP + prior", "Prior given to other learners"),
    ("GBDT + prior", "GBDT + prior", "Prior given to other learners"),
    ("GAT + prior", "GAT + prior", "Prior given to other learners"),
    ("RouteNet-style + prior features", "RouteNet-style + prior features", "Prior given to other learners"),
]


def archive():
    m = pd.read_csv("results/loto_metrics_v3.csv").groupby(["model", "fold"]).mape.mean().groupby("model").mean()
    t = pd.read_csv("results/loto_tests_v3.csv")
    sig = t.assign(w=(t["diff"] < 0) & (t.p_holm < .05)).groupby("model").w.sum()
    return m, sig


FROM_RUNS = {
    "MLP + prior": "{D}/bl_mlp_p_short_*.json",
    "GAT + prior": "{D}/bl_gat_p_short_*.json",
    "QPR-GNN with log-ratio loss": "results/iotfold_log/routenet_qosres_nosched_pf_short_*.json",
}


def from_runs(pattern):
    import glob, json
    rows = [(t, d["seed"], 100 * r["mape"]) for f in glob.glob(pattern) for d in [json.load(open(f))]
            for t, r in d["results"].items() if "(id)" not in t]
    return pd.DataFrame(rows, columns=["topology", "seed", "mape"]).groupby("topology").mape.mean() if rows else None


def sim(D):
    d = pd.read_csv(f"{D}/metrics.csv").groupby(["model", "topology"]).mape.mean().unstack()
    for k, pat in FROM_RUNS.items():
        if k in d.index or (k.startswith("QPR-GNN with log-ratio loss") and "iotfold" not in D):
            continue
        v = from_runs(pat.format(D=D))
        if v is not None:
            d.loc[k] = v.reindex(d.columns)
    o = d.loc[OURS]; res = {}
    for k in d.index:
        v = d.loc[k]
        res[k] = (v.mean(), None if k == OURS else wilcoxon(o, v).pvalue, int((o < v).sum()), len(v))
    return res


def main():
    am, asig = archive(); bn, ns = sim("results/simfold_mape"), sim("results/ns3fold_mape")
    io = sim("results/iotfold") if os.path.exists("results/iotfold/metrics.csv") else {}
    rows = []
    for tab, spec in (("Table 1: published methods", PUBLISHED), ("Table 2: ablation", ABLATION)):
        for disp, key, grp in [("QPR-GNN (ours)", OURS, "Ours")] + spec:
            r = dict(table=tab, group=grp, model=disp, archive=am.get(key, np.nan),
                     archive_sig=None if key == OURS else (f"{int(asig[key])}/3" if key in asig else None))
            for b, R in (("bnnet", bn), ("ns3", ns), ("iot", io)):
                v = R.get(key)
                r[b] = v[0] if v else np.nan
                r[f"{b}_p"] = v[1] if v else None
                r[f"{b}_better"] = f"{v[2]}/{v[3]}" if v and v[1] is not None else None
            rows.append(r)
    df = pd.DataFrame(rows); df.to_csv("results/tables_v3.csv", index=False)

    def cell(x, fmt="{:.2f}"):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)

    def pmark(p):
        return "" if p is None else ("‡" if p < .001 else "†" if p < .05 else " (n.s.)")

    out = ['Table 1  comparison with published methods (state of the art and classical models)\nTable 2  ablation study: the prior alone, each component of QPR-GNN, and the prior given to other learners\n         (matched-information controls)', ""]
    tex = []
    df = df.astype(object).where(df.notna(), None)
    for tab in df.table.unique():
        T = df[df.table == tab]
        out.append(f"{tab}\n{'group':30s}{'model':36s}{'Archive':>10s}{'sig.':>6s}{'BNNetSim':>10s}{'better, p':>18s}{'ns-3':>8s}{'better, p':>18s}{'IoT':>8s}{'better, p':>18s}")
        tex.append(f"% {tab}\n\\begin{{tabular}}{{llrrrr}}\n\\toprule\nGroup & Model & Archive & BNNetSimulator & ns-3 & IoT traffic \\\\\n\\midrule")
        for _, r in T.iterrows():
            bp = "" if r.bnnet_p is None else f"{r.bnnet_better}, {r.bnnet_p:.4f}"
            np_ = "" if r.ns3_p is None else f"{r.ns3_better}, {r.ns3_p:.4f}"
            ip = "" if r.iot_p is None else f"{r.iot_better}, {r.iot_p:.4f}"
            out.append(f"{r.group:30s}{r.model:36s}{cell(r.archive):>10s}{(r.archive_sig or ''):>6s}{cell(r.bnnet):>10s}{bp:>18s}{cell(r.ns3):>8s}{np_:>18s}{cell(r.iot):>8s}{ip:>18s}")
            bold = r.model.startswith("QPR-GNN (ours)")
            f = (lambda s: f"\\textbf{{{s}}}") if bold else (lambda s: s)
            arch_mark = "" if not r.archive_sig else ("$^\\ddagger$" if r.archive_sig == "3/3" else f"$^{{{r.archive_sig}}}$")
            tex.append(f"{r.group} & {f(r.model)} & {f(cell(r.archive))}{arch_mark} & "
                       f"{f(cell(r.bnnet))}{pmark(r.bnnet_p)} & "
                       f"{f(cell(r.ns3))}{pmark(r.ns3_p)} & {f(cell(r.iot))}{pmark(r.iot_p)} \\\\")
        out.append("")
        tex.append("\\bottomrule\n\\end{tabular}\n")
    out.append("sig. = held-out archive topologies where QPR-GNN is significantly better (Holm-corrected).\n"
               "better, p = topologies where QPR-GNN is better, Wilcoxon signed-rank p over the 16 topologies.\n"
               "IoT = cross-topology folds on IoT backhaul traffic, 16 topologies, 3 seeds.\n"
               "* published model with the transmission term added by us. — = not run on that benchmark.")
    os.makedirs("results/analysis", exist_ok=True)
    open("results/analysis/tables_v3.txt", "w").write("\n".join(out) + "\n")
    open("results/analysis/tables_v3.tex", "w").write("\n".join(tex).replace("‡", "$^\\ddagger$").replace("†", "$^\\dagger$")
                                                       .replace(" (n.s.)", "$^{\\mathrm{ns}}$").replace("—", "--") + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
