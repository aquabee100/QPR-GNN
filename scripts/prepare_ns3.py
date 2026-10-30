import glob, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from qprgnn.parse_ns3 import iter_ns3_samples
from qprgnn.data import load_topology

root = sys.argv[1] if len(sys.argv) > 1 else "ns3_inputs"
os.makedirs("cache", exist_ok=True)
strata, done = {}, []
for td in sorted(glob.glob(os.path.join(root, "*", "samples.csv"))):
    td = os.path.dirname(td); T = os.path.basename(td)
    want = len(glob.glob(os.path.join(td, "sample_*.txt")))
    S = list(iter_ns3_samples(td))
    if len(S) < 0.9 * want:
        print(f"skip {T}: incomplete ({len(S)}/{want})"); continue
    name = f"ns3_{T}"
    pickle.dump(S, open(f"cache/{name}.pkl", "wb"), protocol=4)
    for f in glob.glob(f"cache/*_{name}.pkl"):            # derived caches of an older parse
        os.remove(f)
    items, kept = load_topology(name)
    plan = pd.read_csv(os.path.join(td, "samples.csv")).set_index("sample")
    for s in kept:
        k = int(s["graph_file"].split("_")[1].split(".")[0])
        strata[f"{name}|{s['source_file']}|{s['source_line']}"] = dict(sample=k, **{c: (str(v) if isinstance(v, str) else float(v)) for c, v in plan.loc[k].items()})
    done.append(name)
    print(f"{name}: {len(S)} simulated, {len(items)} kept, {items[0]['L']} links, {sum(it['F'] for it in items)} flows")
json.dump(strata, open("cache/ns3_strata.json", "w"))
print("ready:", ",".join(done))
