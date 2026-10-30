import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

from eval_loto import OURS, PREV, PRED, OUT, slug
from qprgnn.protocol import TARGETS
from qprgnn.runs import Topology

SHOW = ["M/M/1/K, aggregate (= our prior alone)", "Gradient-boosted trees", "RouteNet-style (87k)", "RouteNet-Fermi",
        "QT-RouteNet + transmission term", "GAT", "GBDT + prior", "RouteNet-style + prior features", PREV, OURS]
rows = []
for fold in TARGETS:
    items = Topology(fold).items
    meta = np.load(f"{PRED}/meta_{fold}.npz")
    y, tos = meta["y"], meta["tos"]
    txs = np.concatenate([(it["tx"] * (it["path"] >= 0)).sum(1).numpy() for it in items])
    for name in SHOW:
        P = np.load(f"{PRED}/{slug(name)}__{fold}.npz")
        for c in (2, 3, 5):
            D = c * txs
            truth = y > D
            for seed in P.files:
                pred = P[seed] > D
                for cls, m in (("all", np.ones_like(truth)), ("URLLC (ToS 0)", tos == 0)):
                    t, p = truth[m], pred[m]
                    tp, fp, fn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum()
                    rows.append(dict(model=name, fold=fold, c=c, cls=cls, seed=seed, viol_rate=t.mean(),
                                     f1=2 * tp / max(2 * tp + fp + fn, 1), recall=tp / max(t.sum(), 1),
                                     false_alarm=fp / max((~t).sum(), 1)))
df = pd.DataFrame(rows); df.to_csv("results/iot_decision.csv", index=False)
g = df.groupby(["cls", "c", "model"])[["viol_rate", "f1", "recall", "false_alarm"]].mean()
out = ["Deadline-violation detection: budget D_f = c x (flow's own transmission time along its path).",
       "Means over seeds and the 3 held-out topologies (macro). Higher F1/recall and lower false-alarm are better.\n"]
for cls in ("all", "URLLC (ToS 0)"):
    for c in (2, 3, 5):
        out.append(f"--- {cls}, c = {c} (violation rate {100 * g.loc[(cls, c)].viol_rate.iloc[0]:.1f}%) ---")
        out.append(f"{'model':44s} {'F1':>6s} {'recall':>7s} {'false alarm':>12s}")
        for m in SHOW:
            r = g.loc[(cls, c, m)]
            out.append(f"{m:44s} {r.f1:6.3f} {r.recall:7.3f} {r.false_alarm:12.3f}")
        out.append("")
open(f"{OUT}/iot_decision.txt", "w").write("\n".join(out)); print("\n".join(out))
