import json, os
R = [("RouteNet-Fermi, 75 epochs, batch 8 (standard)", "results/loto/fermi_official_short_test_germany50_s0.json"),
     ("RouteNet-Fermi, official schedule", "results/budget/fermi_official_short_test_germany50_s0.json"),
     ("QPR-GNN, log-ratio loss, 25 epochs, batch 8", "results/loto/routenet_qosres_nosched_pf_short_test_germany50_s0.json"),
     ("QPR-GNN, log-ratio loss, official schedule", "results/budget/routenet_qosres_nosched_pf_official_short_test_germany50_s0.json"),
     ("QPR-GNN, 25 epochs, batch 8 (standard)", "results/loto_mape/routenet_qosres_nosched_pf_short_test_germany50_s0.json"),
     ("QPR-GNN, official schedule", "results/budget/routenet_qosres_nosched_pf_official_mape_short_test_germany50_s0.json")]
out = ["Germany50 held out, seed 0. MAPE %: held-out | in-distribution test splits (GEANT, Abilene, Nobel-EU) | best validation"]
for lab, f in R:
    d = json.load(open(f)); r = d["results"]
    out.append(f"{lab:50s} {100 * r['test_germany50']['mape']:6.2f} | " +
               " ".join(f"{100 * r[k]['mape']:5.2f}" for k in ("train_geant(id)", "test_abilene(id)", "test_nobel(id)")) +
               f" | {100 * d['best_val']:5.2f}")
os.makedirs("results/analysis", exist_ok=True)
open("results/analysis/budget.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
