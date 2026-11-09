import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import tables_v3 as T

OURS, FERMI, MSMP = "QPR-GNN (ours)", "RouteNet-Fermi", "MSMP + attention"
PUB = [k for _, k, _ in T.PUBLISHED]
out = []
def say(s): out.append(s)

arch = pd.read_csv("results/loto_metrics_v3.csv").groupby(["model", "fold"]).mape.mean().unstack()
B = {"Archive": arch}
for name, d in (("BNNetSimulator", "results/simfold_mape"), ("ns-3", "results/ns3fold_mape"), ("IoT", "results/iotfold")):
    m = pd.read_csv(f"{d}/metrics.csv").groupby(["model", "topology"]).mape.mean().unstack()
    for k, pat in T.FROM_RUNS.items():
        if k not in m.index and not (k.startswith("QPR-GNN with log-ratio loss") and name != "IoT"):
            v = T.from_runs(pat.format(D=d))
            if v is not None:
                m.loc[k] = v.reindex(m.columns)
    B[name] = m
mean = {b: B[b].mean(axis=1) for b in B}

say("1. Archive per held-out topology, QPR-GNN vs RouteNet-Fermi (MAPE %)")
for t in arch.columns:
    say(f"   {t}: {arch.loc[OURS, t]:.2f} vs {arch.loc[FERMI, t]:.2f}")
say("2. Topology-averaged MAPE and error reduction of QPR-GNN")
for b in B:
    o = mean[b][OURS]
    say(f"   {b}: ours {o:.2f}, Fermi {mean[b][FERMI]:.2f} (-{100*(1-o/mean[b][FERMI]):.0f}%), "
        f"MSMP {mean[b][MSMP]:.2f} (-{100*(1-o/mean[b][MSMP]):.0f}%)")
say("3. Held-out topologies where QPR-GNN is more accurate; median error reduction per benchmark")
for base in (FERMI, MSMP):
    w = n = 0; med = []; lost = []
    for b in B:
        o, v = B[b].loc[OURS], B[b].loc[base]
        w += int((o < v).sum()); n += len(o); med.append(f"{b} {np.median(100*(v-o)/v):.0f}%")
        lost += [f"{t} ({o[t]:.2f} vs {v[t]:.2f}, gap {o[t]-v[t]:.2f})" for t in o.index[o >= v]]
    say(f"   vs {base}: {w} of {n}; medians: " + ", ".join(med))
    if lost: say("   remaining topologies: " + "; ".join(lost))
say("4. Per-topology sign-flip tests (Holm) against the 14 published methods")
tw = tl = tn = 0
for f in ("results/loto_tests_v3.csv", "results/simfold_mape/tests.csv", "results/ns3fold_mape/tests.csv", "results/iotfold/tests.csv"):
    t = pd.read_csv(f); pc = [c for c in t.columns if "holm" in c.lower()][0]; t = t[t.model.isin(PUB)]
    tw += int(((t["diff"] < 0) & (t[pc] < .05)).sum()); tl += int(((t["diff"] > 0) & (t[pc] < .05)).sum()); tn += len(t)
say(f"   {tw} significant improvements in {tn} comparisons; {tl} significant losses")
say("5. Wilcoxon across the 16 topologies against each published method (fewest wins, largest p)")
from scipy.stats import wilcoxon
for b in ("BNNetSimulator", "ns-3", "IoT"):
    o = B[b].loc[OURS]; r = [(int((o < B[b].loc[m]).sum()), wilcoxon(o, B[b].loc[m]).pvalue) for m in PUB]
    say(f"   {b}: at least {min(x for x, _ in r)} of 16 topologies, p <= {max(p for _, p in r):.1e}")
say("6. Prior alone (M/M/1/K) relative to QPR-GNN")
pr = [mean[b]["M/M/1/K, aggregate (= our prior alone)"] / mean[b][OURS] for b in B]
say(f"   {min(pr):.1f} to {max(pr):.1f} times the error of QPR-GNN")
say("7. Ablation: error averaged over the four benchmarks")
for disp, k, _ in [("QPR-GNN (ours)", OURS, "")] + T.ABLATION:
    if all(k in mean[b] for b in B):
        say(f"   {disp:36s} {np.mean([mean[b][k] for b in B]):.2f}")
say("8. IoT: QPR-GNN against every learner given the prior")
ctl = ["MLP + prior", "GBDT + prior", "GAT + prior", "RouteNet-style + prior features"]
o = B["IoT"].loc[OURS]; r = [(int((o < B["IoT"].loc[m]).sum()), wilcoxon(o, B["IoT"].loc[m]).pvalue) for m in ctl]
say(f"   better on {min(x for x, _ in r)} to {max(x for x, _ in r)} of 16 topologies, p <= {max(p for _, p in r):.3f}")
say("9. ns-3: QPR-GNN across training seeds, and GAT + prior")
def seeds(pat):
    rows = [(d["seed"], t, 100 * v["mape"]) for f in glob.glob(pat) for d in [json.load(open(f))]
            for t, v in d["results"].items() if "(id)" not in t]
    return pd.DataFrame(rows, columns=["s", "t", "m"]).groupby("s").m.mean()
s = seeds("results/ns3fold_mape/routenet_qosres_nosched_pf_short_*.json")
say(f"   QPR-GNN seeds {s.min():.2f}-{s.max():.2f}; mean {mean['ns-3'][OURS]:.2f}; GAT + prior {mean['ns-3']['GAT + prior']:.2f} "
    f"(gap {mean['ns-3'][OURS]-mean['ns-3']['GAT + prior']:.2f})")
say("10. Training time of QPR-GNN on IoT traffic (median over 12 runs, CPU)")
tt = [json.load(open(f))["train_time_s"] for f in glob.glob("results/iotfold/routenet_qosres_nosched_pf_short_*.json")]
say(f"   {np.median(tt)/60:.0f} minutes")
say("11. Pooled archive percentiles of absolute percentage error")
c = pd.read_csv("results/ape_cdf_v3.csv", index_col=0)
for q in (90, 95):
    say(f"   P{q}: ours {c.loc[float(q), OURS]:.2f}, Fermi {c.loc[float(q), FERMI]:.2f}")
say("12. IoT MAPE by traffic class (ours vs best published)")
C = pd.read_csv("results/iotfold/metrics_by_tos.csv").groupby(["model", "tos"])[["ape_sum", "n"]].sum()
cm = (100 * C.ape_sum / C.n).unstack()
for tos, lab in ((0, "control/alarm"), (1, "telemetry"), (2, "video")):
    bp = cm.loc[[m for m in PUB if m in cm.index], tos]
    say(f"   {lab}: ours {cm.loc[OURS, tos]:.2f}, best published {bp.min():.2f} ({bp.idxmin()})")
say("13. Analytic models and generic graph learners on IoT traffic")
g = ["M/M/1 (per-flow packet size)", "M/M/1/K, aggregate (= our prior alone)", "M/G/1 (Pollaczek-Khinchine)",
     "MLP", "Gradient-boosted trees", "GCN", "GAT", "Graph transformer (GPS-style)"]
say(f"   {min(mean['IoT'][m] for m in g):.0f}-{max(mean['IoT'][m] for m in g):.0f}%")
say("14. Delay-budget violation detection (results/analysis/iot_decision_v3.txt)")
txt = open("results/analysis/iot_decision_v3.txt").read()
def row(sec, model):
    blk = txt.split(sec)[1].split("\n---")[0]
    return [float(x) for x in re.search(re.escape(model) + r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", blk).groups()]
o3, f3 = row("all, c = 3", OURS), row("all, c = 3", FERMI)
o5, f5 = row("all, c = 5", OURS), row("all, c = 5", FERMI)
say(f"   c=3 recall: ours {100*o3[1]:.1f}%, Fermi {100*f3[1]:.1f}% ({100*(o3[1]/f3[1]-1):.0f}% more violations detected)")
say(f"   c=5 F1: ours {o5[0]:.3f}, Fermi {f5[0]:.3f} ({o5[0]/f5[0]:.1f} times)")
say("15. Inference cost (results/analysis/efficiency_v3.txt)")
e = open("results/analysis/efficiency_v3.txt").read()
say("   " + [l for l in e.splitlines() if l.startswith("ns-3")][0])
say("   " + [l for l in e.splitlines() if l.startswith(OURS)][0])
open("results/analysis/paper_numbers.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
