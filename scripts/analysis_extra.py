import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch

from eval_loto import PRED, OUT, OURS, slug
from qprgnn.protocol import TARGETS
from qprgnn.runs import Topology

MODELS = ["M/M/1/K, aggregate (= our prior alone)", "Gradient-boosted trees", "GAT", "RouteNet-style (87k)",
          "RouteNet-Fermi", "QT-RouteNet + transmission term", "GBDT + prior", "RouteNet-style + prior features", OURS]
LOAD_BINS = [0, 0.3, 0.5, 0.7, 0.85, 10]
MIS_BINS = [0, 0.1, 0.3, 0.6, 1.0, 10]


def flow_descriptors(topo):
    load, mis = [], []
    for it in topo.items:
        m = it["path"] >= 0; pc = it["path"].clamp(min=0)
        rho = it["link_x"][:, 0][pc] * m                     # offered utilisation of each hop (input side)
        load.append(rho.max(1).values.numpy())
        lr = torch.log(it["hop_ratio"].clamp(min=1e-6)).abs() * m
        mis.append(lr.max(1).values.numpy())
    return np.concatenate(load), np.concatenate(mis)


def main():
    rows = []
    for fold in TARGETS:
        topo = Topology(fold)
        load, mis = flow_descriptors(topo)
        for m in MODELS:
            P = np.load(f"{PRED}/{slug(m)}__{fold}.npz")
            ape = np.mean([np.abs(P[k] - topo.y) / topo.y for k in P.files], 0)
            rows.append(pd.DataFrame(dict(model=m, fold=fold, ape=100 * ape, load=load, mis=mis)))
    d = pd.concat(rows)
    out = []
    for col, bins, lab, fn in (("load", LOAD_BINS, "max offered utilisation along the path", "error_by_load"),
                               ("mis", MIS_BINS, "packet-size mismatch max_k |log(s_f / s_bar)| along the path", "error_by_size_mismatch")):
        d["bin"] = pd.cut(d[col], bins, right=False)
        g = d.groupby(["bin", "model"], observed=True).ape.mean().unstack()[MODELS]
        n = d[d.model == OURS].groupby("bin", observed=True).size()
        txt = [f"Held-out MAPE (%) by {lab}; 3 archive targets pooled; seed-averaged single-model errors.",
               f"{'bin':16s}{'flows':>8s}" + "".join(f"{m[:14]:>16s}" for m in MODELS)]
        for b in g.index:
            txt.append(f"{str(b):16s}{n[b]:8d}" + "".join(f"{g.loc[b, m]:16.2f}" for m in MODELS))
        open(f"{OUT}/{fn}.txt", "w").write("\n".join(txt) + "\n"); out += txt + [""]
        g.assign(flows=n).to_csv(f"results/{fn}.csv")
    q = np.linspace(0, 100, 201)
    cdf = pd.DataFrame({m: np.percentile(d[d.model == m].ape, q) for m in MODELS}, index=q)
    cdf.index.name = "percentile"; cdf.to_csv("results/ape_cdf.csv")
    print("\n".join(out))


if __name__ == "__main__":
    main()
