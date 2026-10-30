import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from qprgnn.parse import load_graph, load_routing

out = []
zero = act = bad_paths = 0
for root in sorted(glob.glob("sim_inputs/*/")):
    for g in sorted(glob.glob(root + "graphs/graph_*.txt"))[:60]:
        G, links, cap, lidx, _, node = load_graph(g)
        for n in range(len(node["nq"])):
            q = int(node["nq"][n]); act += q; zero += int((node["buffers"][n, :q] <= 0).sum())
    sim = open(root + "simulation.txt").read().split()
    for line in sim[:4]:
        gp, rp, _ = line.split(",")
        paths = load_routing(root + rp, root + gp)
        G, links, cap, lidx, _, _ = load_graph(root + gp)
        bad_paths += sum(1 for p in paths.values() for a, b in zip(p[:-1], p[1:]) if (a, b) not in lidx)
S = pd.concat([pd.read_csv(f).assign(topology=f.split("/")[1]) for f in sorted(glob.glob("sim_inputs/*/samples.csv"))])
out.append(f"active queue slots with zero/negative parsed buffer: {zero} / {act}")
out.append(f"routing hops not present as graph links (first 4 samples per topology): {bad_paths}")
out.append(f"samples: {len(S)} over {S.topology.nunique()} topologies")
out.append("\nper stratum: samples, max directional link utilisation (median / 5th / 95th pct), median link util")
for st, g in S.groupby("stratum"):
    out.append(f"  {st:16s} {len(g):5d}  max-util {g.max_util.median():.3f} / {g.max_util.quantile(.05):.3f} / {g.max_util.quantile(.95):.3f}"
               f"   median-util {g.median_util.median():.3f}")
out.append(f"\nsamples with max utilisation >= 0.85: {(S.max_util >= 0.85).sum()}; >= 1.0: {(S.max_util >= 1.0).sum()}")
open("results/analysis/sim_inputs_audit.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
