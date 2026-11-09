import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

rows = []
if os.path.exists("cache/train_geant.pkl"):
    from qprgnn.data import load_topology
    for name, lab in (("train_geant", "GEANT"), ("test_abilene", "Abilene"), ("test_nobel", "Nobel-EU"),
                      ("test_germany50", "Germany50")):
        _, S = load_topology(name)
        rows.append(dict(dataset="RouteNet-Fermi archive", topology=lab, nodes=str(S[0]["n_nodes"]),
                         states=len(S), flows=int(sum(len(s["y_delay"]) for s in S))))
topo = pd.read_csv("results/sim_topologies.csv"); topo = topo[topo.kind == "cross"]
nodes = f"{topo.N.min()}-{topo.N.max()}"
for d, lab in (("results/simfold_mape", "BNNetSimulator"), ("results/ns3fold_mape", "ns-3"),
               ("results/iotfold", "ns-3, IoT traffic")):
    st = fl = 0
    for f in glob.glob(f"{d}/preds_*.npz"):
        n = np.load(f)["flows"]; st += len(n); fl += int(n.sum())
    if st:
        rows.append(dict(dataset=lab, topology="16 Topology Zoo networks", nodes=nodes, states=st, flows=fl))
df = pd.DataFrame(rows)
if os.path.exists("results/dataset_stats.csv"):
    old = pd.read_csv("results/dataset_stats.csv", dtype={"nodes": str})
    have = set(df.dataset) if len(df) else set()
    df = pd.concat([old[~old.dataset.isin(have)], df])
    order = ["RouteNet-Fermi archive", "BNNetSimulator", "ns-3", "ns-3, IoT traffic"]
    df = df.assign(o=df.dataset.map(order.index)).sort_values("o", kind="stable").drop(columns="o")
df.to_csv("results/dataset_stats.csv", index=False)
held = df[(df.dataset == "RouteNet-Fermi archive") & (df.topology != "GEANT")].flows.sum()
txt = df.to_string(index=False) + f"\n\nheld-out archive flows (Abilene + Nobel-EU + Germany50): {held}\n"
open("results/analysis/dataset_stats.txt", "w").write(txt); print(txt)
