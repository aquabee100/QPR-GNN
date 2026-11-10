import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from qprgnn.parse_ns3 import iter_ns3_samples

out = []
for root, lab in (("ns3_inputs", "ns-3"), ("iot_inputs", "ns-3, IoT traffic")):
    rows = []
    for td in sorted(glob.glob(f"{root}/*/samples.csv")):
        td = os.path.dirname(td)
        plan = pd.read_csv(f"{td}/samples.csv").set_index("sample")
        for s in iter_ns3_samples(td):
            k = int(s["graph_file"].split("_")[1].split(".")[0])
            y = np.asarray(s["y_delay"])
            rows.append(dict(topology=os.path.basename(td), sample=k, stratum=plan.loc[k, "stratum"],
                             excluded=bool((y <= 0).any()), flows=len(y), silent=int((y <= 0).sum())))
    d = pd.DataFrame(rows)
    out.append(f"{lab}: {len(d)} simulated states, {int(d.excluded.sum())} excluded (a flow received no packet)")
    t = d.groupby("stratum").agg(states=("excluded", "size"), excluded=("excluded", "sum"))
    t["excluded_%"] = 100 * t.excluded / t.states
    out.append(t.round(2).to_string())
    e = d[d.excluded]
    if len(e):
        out.append(f"silent flows in excluded states: {int(e.silent.sum())} of {int(e.flows.sum())} "
                   f"({100 * e.silent.sum() / e.flows.sum():.2f}%); median per state {int(e.silent.median())}")
        out.append("excluded states per topology: " + ", ".join(f"{k} {v}" for k, v in e.groupby("topology").size().items()))
    out.append("")
os.makedirs("results/analysis", exist_ok=True)
open("results/analysis/excluded_states.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
