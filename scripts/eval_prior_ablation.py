import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import eval_loto as E
from qprgnn.protocol import TARGETS
from qprgnn.runs import Topology

ROWS = {  # readout base only, no prior features
    "transmission time only, tx (no queueing prior)": ("qprgnn", "routenet_tx_res"),
    "tx / (1 - rho)  (M/M/1 on the flow's packet size)": ("qprgnn", "routenet"),
    "tx + FIFO M/M/1 waiting (infinite buffer)": ("qprgnn", "routenet_qosres_mm1"),
    "tx + QoS-aware M/M/1/K waiting (SP/WFQ/DRR)": ("qprgnn", "routenet_qosres"),
    "tx + FIFO M/M/1/K waiting (our prior)": ("qprgnn", "routenet_qosres_nosched"),
}
out = ["Prior ablation: per-hop readout base = prior x exp(learned correction); same backbone (25k), inputs and",
       "protocol. Held-out MAPE % (single model, mean ± std over 5 seeds).\n",
       f"{'per-hop prior':52s}" + "".join(f"{f:>18s}" for f in TARGETS) + f"{'avg':>8s}"]
res = {n: [] for n in ROWS}
for f in TARGETS:
    topo = Topology(f)
    for n, (kind, key) in ROWS.items():
        P = E.all_preds("ablation " + n, kind, key, f, topo)
        res[n].append([E.scores(p, topo.y)["mape"] for p in P.values()])
for n in ROWS:
    out.append(f"{n:52s}" + "".join(f"{np.mean(v):9.3f} ± {np.std(v, ddof=1):5.3f}" for v in res[n])
               + f"{np.mean([np.mean(v) for v in res[n]]):8.3f}")
import pandas as pd
pd.DataFrame([dict(prior=n, fold=f, seed=k, mape=v) for n in ROWS for f, vs in zip(TARGETS, res[n]) for k, v in enumerate(vs)]
             ).to_csv("results/prior_ablation.csv", index=False)
open(f"{E.OUT}/prior_ablation.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
