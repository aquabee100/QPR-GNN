import os, pickle, re, sys, tarfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
os.makedirs("results/analysis", exist_ok=True)

out = []
for t in ["train/geant", "test/abilene", "test/nobel", "test/germany50"]:
    d = "data_traces/" + t
    arch = sorted(f for f in os.listdir(d) if re.match(r"results_.*\.tar\.gz$", f))
    lines = unstable = incomplete = 0
    for f in arch:
        with tarfile.open(os.path.join(d, f)) as tar:
            names = tar.getnames(); base = names[0].split("/")[0]
            need = [f"{base}/{n}" for n in ("simulationResults.txt", "traffic.txt", "stability.txt", "input_files.txt")]
            if not all(n in names for n in need):
                incomplete += 1; continue
            sta = tar.extractfile(f"{base}/stability.txt").read().decode().splitlines()
            lines += len(sta); unstable += sum(";OK;" not in s for s in sta)
    S = pickle.load(open(f"cache/{t.replace('/', '_')}.pkl", "rb"))
    nonpos = [s for s in S if not (len(s["y_delay"]) and np.all(s["y_delay"] > 0))]
    zf = [int((s["y_delay"] <= 0).sum()) for s in nonpos]
    zl = [float(s["y_drop"][s["y_delay"] <= 0].mean()) if len(s["y_drop"]) else np.nan for s in nonpos]
    out.append(f"{t:16s} archives {len(arch):4d} (incomplete {incomplete}) | sample lines {lines:5d} | "
               f"not stable {unstable:3d} | parsed {len(S):4d} | dropped: a flow with delay <= 0 {len(nonpos):3d} "
               f"| kept {len(S) - len(nonpos):4d}")
    if nonpos:
        out.append(f"{'':16s} excluded scenarios: flows with delay <= 0 per scenario {sorted(zf)[:10]}...; "
                   f"mean drop count of those flows {np.nanmean(zl):.1f} (same rule as the official generator)")
open("results/analysis/attrition.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
