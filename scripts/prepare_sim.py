import glob, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from qprgnn.parse import iter_samples
from qprgnn.data import load_topology

root = sys.argv[1] if len(sys.argv) > 1 else "sim_inputs"
os.makedirs("cache", exist_ok=True)
strata, done = {}, []
for td in sorted(glob.glob(os.path.join(root, "*", "simulation.txt"))):
    td = os.path.dirname(td); T = os.path.basename(td)
    res = os.path.join(td, "results", T)
    if not glob.glob(os.path.join(res, "*.tar.gz")):
        continue
    want = sum(1 for _ in open(os.path.join(td, "simulation.txt")))
    S = list(iter_samples(res))
    # accept topologies with >= 90% samples
    if len(S) < 0.9 * want:
        print(f"skip {T}: incomplete ({len(S)}/{want})"); continue
    name = f"sim_{T}"
    pickle.dump(S, open(f"cache/{name}.pkl", "wb"), protocol=4)
    items, kept = load_topology(name)
    plan = pd.read_csv(os.path.join(td, "samples.csv")).set_index("sample")
    for s in kept:
        k = int(s["graph_file"].split("_")[1].split(".")[0])
        strata[f"{name}|{s['source_file']}|{s['source_line']}"] = dict(sample=k, **{c: (str(v) if isinstance(v, str) else float(v)) for c, v in plan.loc[k].items()})
    done.append(name)
    print(f"{name}: {len(S)} parsed, {len(items)} kept, {items[0]['L']} links, {sum(it['F'] for it in items)} flows")
json.dump(strata, open("cache/sim_strata.json", "w"))
print("ready:", ",".join(done))
