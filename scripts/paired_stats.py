import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import tables_v3 as T

OURS = T.OURS
rng = np.random.default_rng(0)
arch = pd.read_csv("results/loto_tests_v3.csv")
B = {}
for name, d in (("BNNetSimulator", "results/simfold_mape"), ("ns-3", "results/ns3fold_mape"), ("IoT traffic", "results/iotfold")):
    m = pd.read_csv(f"{d}/metrics.csv").groupby(["model", "topology"]).mape.mean().unstack()
    for k, pat in T.FROM_RUNS.items():
        if k not in m.index and not (k.startswith("QPR-GNN with log-ratio loss") and name != "IoT traffic"):
            v = T.from_runs(pat.format(D=d))
            if v is not None:
                m.loc[k] = v.reindex(m.columns)
    B[name] = m
rows = []
for disp, key, grp in [(a, b, c) for a, b, c in T.PUBLISHED] + [(a, b, "Ablation") for a, b, _ in T.ABLATION if b != T.PUBLISHED[1][1]]:
    r = dict(group="Baseline" if grp != "Ablation" else "Ablation", model=disp)
    a = arch[arch.model == key]
    if len(a):
        r["Archive"] = f"{a['diff'].mean():+.2f} [{a.lo.min():+.2f}, {a.hi.max():+.2f}] {int(((a['diff'] < 0) & (a.p_holm < .05)).sum())}/3"
    for b, m in B.items():
        if key not in m.index:
            continue
        dlt = (m.loc[OURS] - m.loc[key]).values
        bs = [rng.choice(dlt, len(dlt)).mean() for _ in range(10000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        r[b] = f"{dlt.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] {int((dlt < 0).sum())}/16 p={wilcoxon(dlt).pvalue:.1e}"
    rows.append(r)
D = pd.DataFrame(rows).fillna("")
D.to_csv("results/paired_stats.csv", index=False)
head = ("Paired MAPE difference of QPR-GNN minus the row (percentage points; negative = QPR-GNN more accurate).\n"
        "Archive: mean over the 3 held-out topologies [lowest lower and highest upper bound of the per-topology 95%\n"
        "cluster-bootstrap intervals], held-out topologies where QPR-GNN is significantly better (sign-flip test, Holm).\n"
        "Simulator and IoT studies: mean over the 16 held-out topologies [95% bootstrap interval over topologies],\n"
        "topologies where QPR-GNN is better, Wilcoxon signed-rank p over the 16 topologies. Seeds: 5 (archive), 3 (others).\n\n")
open("results/analysis/paired_stats.txt", "w").write(head + D.to_string(index=False) + "\n")
print(head + D.to_string(index=False))
