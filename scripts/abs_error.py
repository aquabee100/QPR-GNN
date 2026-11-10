import glob, os
import numpy as np, pandas as pd

MODELS = {"QPR-GNN (ours)": ("QPR_GNN__ours", "QPR-GNN_(ours)"), "RouteNet-Fermi": ("RouteNet_Fermi", "RouteNet-Fermi"),
          "MSMP + attention": ("MSMP___attention", "MSMP_+_attention")}
rows = []
for t in ("abilene", "nobel", "germany50"):
    y = np.load(f"results/preds_v3/meta_test_{t}.npz")["y"]
    for m, (a, _) in MODELS.items():
        P = np.load(f"results/preds_v3/{a}__test_{t}.npz")
        for s in P.files:
            rows.append(dict(benchmark="Archive", topology=t, model=m, seed=s, mean_delay=1e3 * y.mean(),
                             mae=1e3 * np.abs(P[s] - y).mean(), medae=1e3 * np.median(np.abs(P[s] - y))))
for d, b in (("results/simfold_mape", "BNNetSimulator"), ("results/ns3fold_mape", "ns-3"), ("results/iotfold", "IoT traffic")):
    for f in sorted(glob.glob(f"{d}/preds_*.npz")):
        P = np.load(f); y = P["y"]
        for m, (_, k) in MODELS.items():
            for key in [x for x in P.files if x.startswith(k + "|")]:
                rows.append(dict(benchmark=b, topology=os.path.basename(f)[6:-4], model=m, seed=key.split("|")[1],
                                 mean_delay=1e3 * y.mean(), mae=1e3 * np.abs(P[key] - y).mean(),
                                 medae=1e3 * np.median(np.abs(P[key] - y))))
D = pd.DataFrame(rows); D.to_csv("results/abs_error.csv", index=False)
T = D.groupby(["benchmark", "model"])[["mean_delay", "mae", "medae"]].mean()
txt = ("Absolute delay error (ms): mean over held-out topologies and seeds of the per-topology mean (mae) and\n"
       "median (medae) absolute error; mean_delay = mean simulated delay of the held-out flows.\n\n" + T.to_string(float_format=lambda v: f"{v:.3f}") + "\n")
open("results/analysis/abs_error.txt", "w").write(txt); print(txt)
