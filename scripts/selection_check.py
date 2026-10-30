import glob, json
import numpy as np

C = {"routenet_qosres_nosched_pf": "prior as readout base + input features (FINAL)",
     "routenet_qosres_nosched": "prior as readout base only",
     "routenet_txres_pf": "prior as input features only (tx readout)",
     "routenet_qosres": "QoS-aware M/M/1/K prior as readout base",
     "routenet_qosres_mm1": "FIFO M/M/1 prior as readout base",
     "routenet_fermi": "Fermi-style readout tx (1 + softplus r)",
     "routenet_h60": "RouteNet-style 87k (M/M/1 readout)",
     "routenet": "RouteNet-style 25k (M/M/1 readout)"}
out = ["Mean best validation MAPE % on the validation splits of the TRAINING topologies",
       "(3 LOTO folds x seeds 0-2; lower is better; no held-out data involved)"]
for k, lab in C.items():
    v = [json.load(open(f))["best_val"] * 100 for f in glob.glob(f"results/loto/{k}_short_test_*_s[012].json")]
    if v:
        out.append(f"{lab:52s} {np.mean(v):6.3f}  (n={len(v)})")
open("results/analysis/selection.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
