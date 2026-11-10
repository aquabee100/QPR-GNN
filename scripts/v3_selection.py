import glob, json
import numpy as np

def mean_val(pat):
    return np.mean([json.load(open(f))["best_val"] * 100 for f in glob.glob(pat)])


V1 = mean_val("results/loto/routenet_qosres_nosched_pf_short_test_*_s[012].json")
CTL = mean_val("results/size_control/sizectl_h34_short_test_*_s[012].json")
P_CTL = json.load(open(glob.glob("results/size_control/*.json")[0]))["params"]
C = {"v3_D": "D: decoupled residual (tx * exp(a) + W * exp(b))",
     "v3_M": "M: MAPE loss",
     "v3_DM": "DM: decoupled residual trained on MAPE"}
res = {}
out = ["Mean best validation MAPE % on the validation splits of the TRAINING topologies",
       "(3 leave-one-topology-out folds x seeds 0-2; no held-out data involved)",
       f"{'log-ratio loss (reference)':52s} {V1:6.3f}", f"{'log-ratio loss at H=34, ' + str(P_CTL) + ' parameters (size control)':52s} {CTL:6.3f}"]
for k, lab in C.items():
    v = [json.load(open(f))["best_val"] * 100 for f in glob.glob(f"results/v3/{k}_short_test_*_s[012].json")]
    if v:
        res[k] = np.mean(v)
        out.append(f"{lab:52s} {res[k]:6.3f}  (n={len(v)})")
ok = {k: v for k, v in res.items() if v < V1 and v < CTL and len(glob.glob(f"results/v3/{k}_short_test_*_s[012].json")) == 9}
out.append("")
out.append(f"decision: adopt {C[min(ok, key=ok.get)]}" if ok else "decision: no candidate qualifies; keep log-ratio loss")
open("results/analysis/v3_selection.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
