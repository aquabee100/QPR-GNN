import glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch

from qprgnn.runs import Topology, load_model, file_key
from qprgnn.data import collate
from qprgnn.baselines import collate_graph, flow_table, prior_table, prior_sum
from qprgnn.fermi import collate_fermi
from qprgnn.qtroutenet import collate_qt

torch.set_num_threads(1)
MODELS = {"QPR-GNN (ours)": ("qprgnn", "routenet_qosres_nosched_pf"), "RouteNet-style (87k)": ("qprgnn", "routenet_h60"),
          "RouteNet-Fermi": ("fermi", "fermi_official"), "QT-RouteNet + transmission term": ("qt", "qtroutenet_tx"),
          "GAT": ("baseline", "gat"), "GBDT + prior": ("baseline", "gbdt_p")}
if os.environ.get("V3"):
    MODELS["MSMP + attention"] = ("msmp", "msmp_attn")


@torch.no_grad()
def time_one(kind, key, m, topo, i):
    if kind == "qprgnn":
        b = collate([topo.items[i]]); f = lambda: m(b)
    elif kind == "fermi":
        b = collate_fermi([topo.fermi[i]]); f = lambda: m(b)
    elif kind == "qt":
        b = collate_qt([topo.qt[i]]); f = lambda: m(b)
    elif kind == "msmp":
        from qprgnn.msmp import collate_msmp
        b = collate_msmp([topo.fermi[i]]); f = lambda: m(b)
    elif key.startswith("gbdt"):
        it = topo.items[i]; X = torch.cat([flow_table(it), prior_table(it)], 1).numpy(); f = lambda: m.predict(X)
    else:
        b = collate_graph([topo.items[i]], prior=key.endswith("_p")); f = lambda: m(b)
    f(); t = time.perf_counter(); f(); return time.perf_counter() - t


V3 = bool(os.environ.get("V3"))
SUF = "_v3" if V3 else ""


def main():
    rows = []
    for tname, run_dir0, pat in (("ns3_Palmetto", "results/ns3fold", "f0"), ("test_germany50", "results/loto", "test_germany50")):
        topo = Topology(tname)
        idx = list(range(len(topo.items)))[:60]
        for name, (kind, key) in MODELS.items():
            run_dir = run_dir0.replace("fold", "fold_mape").replace("loto", "loto_mape") if V3 and name == "QPR-GNN (ours)" else run_dir0
            j = sorted(glob.glob(f"{run_dir}/{file_key(kind, key)}_short_{pat}_s0.json"))[0]
            d, m = load_model(kind, key, j, topo.items[0])
            params = d.get("params")
            ts = [time_one(kind, key, m, topo, i) for i in idx]
            rows.append(dict(topology=tname, model=name, params=params, median_ms=1000 * np.median(ts), p90_ms=1000 * np.quantile(ts, .9),
                             flows=np.median([topo.items[i]["F"] for i in idx])))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"results/efficiency{SUF}.csv", index=False)
    sim = pd.read_csv("ns3_inputs/Palmetto/timing.log", sep=" ", header=None, names=["sample", "s", "status"])
    out = ["Prediction cost per sample (CPU, 1 thread; median over samples) vs ns-3 packet-level simulation.",
           f"ns-3 (lab server, default build), Palmetto: median {sim.s.median():.0f} s per sample, max {sim.s.max():.0f} s\n",
           f"{'model':34s}{'params':>9s}{'size KB':>9s}{'Palmetto ms':>13s}{'Germany50 ms':>14s}{'speed-up vs ns-3':>18s}"]
    for name in MODELS:
        a = df[(df.model == name) & (df.topology == "ns3_Palmetto")].iloc[0]; b = df[(df.model == name) & (df.topology == "test_germany50")].iloc[0]
        out.append(f"{name:34s}{int(a.params):9d}{4 * a.params / 1024:9.0f}{a.median_ms:13.1f}{b.median_ms:14.1f}{sim.s.median() * 1000 / a.median_ms:17.0f}x")
    open(f"results/analysis/efficiency{SUF}.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))


if __name__ == "__main__":
    main()
