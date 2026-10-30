# Results: QPR-GNN, queueing-prior residual graph learning for transferable delay prediction

Every number below comes from a file under `results/`; each section names its source. Status and
Theory: `docs/THEORY.md`.

## 1. Method (run name `routenet_qosres_nosched_pf`; `qprgnn/model.py`)
* **Backbone:** RouteNet-style flow ↔ link message passing; H=32, T=4 iterations, 25k parameters.
* **Analytic prior per hop** (`qprgnn/queueing.py`):
  - the flow's own transmission time s_f/C;
  - plus the port's FIFO M/M/1/K waiting time W, computed from the aggregate traffic on the port,
    with packet-weighted mean packet size s̄ and buffer K = B/s̄ packets. It is closed form and
    finite for ρ ≥ 1.
* **The prior is used twice:**
  1. *Readout base:* per-hop delay = (s_f/C + W) · e^r, with a zero-initialised last layer, so
     training starts at the analytic prior.
  2. *Input features* on every hop: log(1 + W/tx) and log(s_f/s̄).
* **Loss:** |log ŷ − log y|. Adam 2e-3 with cosine schedule; batch 8; 25 epochs.
* **No outcome quantity enters any input:** delay, utilisation, queue occupancy, jitter and drops
  are all excluded. Rates and packet counts are the generated (offered) traffic, the same fields
  the official RouteNet-Fermi generator uses.
* **Selection on validation data only.** Mean best validation MAPE over 9 runs, from
  `results/analysis/selection.txt`:

  | variant | validation MAPE |
  |---|---|
  | prior as readout base + input features (final) | 5.18 |
  | prior as input features only | 5.19 |
  | prior as readout base only | 5.73 |
  | RouteNet-style (M/M/1 readout) | 6.43 |

## 2. Protocol (`qprgnn/protocol.py`)
* **Data:** RouteNet-Fermi real-traffic dataset (GEANT, Abilene, Nobel-EU, Germany50). Attrition:
  `results/analysis/attrition.txt`. Scenarios with any flow delay ≤ 0 are dropped (the official
  rule): 24 GEANT and 2 Germany50 scenarios.
* **Leave-one-topology-out:**
  - Train on GEANT plus two other archive topologies, test on the complete third.
  - Each training topology has an archive-level 80/10/10 split and at most 250 training samples.
  - GEANT is always a source, because its train and test archives are identical.
  - So there are three held-out targets.
* **Equal protocol for every learned model:** same splits, cap, batch, validation-only
  selection. RouteNet-Fermi and QT-RouteNet get 75 epochs (3×) and their own loss and
  optimiser.
* **Statistics** (`scripts/eval_loto.py`):
  - Single-model, flow-weighted MAPE on the complete held-out set: errors are computed per seed,
    then averaged over 5 seeds.
  - Paired comparison of ours vs each model with the same statistic; resampling unit = source
    archive. The 95% CI comes from a cluster bootstrap; the p-value from a sign-flip
    randomisation test, with Holm correction over all tests.
* **The three archive targets were seen during development,** so they are development
  benchmarks. The confirmatory test is §6.

## 3. Main result (`results/analysis/loto_table.txt`, `loto_significance.txt`; `figures/fig_loto_main.pdf`, `fig_loto_all_models.pdf`)
Held-out MAPE (%), mean over 5 seeds (± std for the strongest models):

| group | model | Abilene | Nobel-EU | Germany50 | avg |
|---|---|---|---|---|---|
| classical analytic | M/M/1 (per-flow packet size) | 7.43 | 17.74 | 11.19 | 12.12 |
| classical analytic | M/M/1/K, aggregate (= our prior alone) | 6.66 | 16.49 | 10.11 | 11.09 |
| classical analytic | M/G/1 (Pollaczek-Khinchine) | 10.22 | 20.43 | 13.01 | 14.55 |
| classical ML | MLP | 5.07 | 11.77 | 11.73 | 9.52 |
| classical ML | Gradient-boosted trees | 5.10 | 11.56 | 11.65 | 9.44 |
| current GNN | RouteNet-style (25k) | 3.86 | 9.49 | 9.17 | 7.51 |
| current GNN | RouteNet-style (87k) | 3.94 | 9.22 | 9.00 | 7.39 |
| latest GNN | RouteNet-Fermi | 3.64 ± 0.16 | 8.83 ± 0.32 | 9.32 ± 0.10 | 7.27 |
| queueing-informed GNN | QT-RouteNet (published design) | 25.46 | 33.10 | 41.19 | 33.25 |
| queueing-informed GNN | QT-RouteNet + transmission term | 5.04 | 11.47 | 10.32 | 8.94 |
| modern GNN | GCN / GAT / graph transformer (GPS-style) | 4.95 / 5.11 / 5.83 | 11.78 / 11.94 / 11.99 | 11.35 / 11.12 / 12.36 | 9.36 / 9.39 / 10.06 |
| matched information | MLP + prior | 3.25 | 9.23 | 8.72 | 7.07 |
| matched information | GBDT + prior | 3.49 | 8.66 | **8.45** | 6.86 |
| matched information | GAT + prior | 3.05 | 9.39 | 8.62 | 7.02 |
| matched information | RouteNet-style + prior features | 3.09 | 8.04 | 8.73 | 6.62 |
| ablation | ours, prior as readout base only | 2.83 | 8.15 | 8.46 | 6.48 |
| **ours** | **QPR-GNN** | **2.78 ± 0.06** | **7.89 ± 0.44** | 8.49 ± 0.33 | **6.39** |

* **Other metrics (avg of 3 topologies):** ours is best on all of them.

  | metric | ours | best other model |
  |---|---|---|
  | median APE | **2.59** | 2.68 (ablation); 2.69 (GBDT + prior) |
  | 90th-percentile APE | **15.54** | 15.67 (ablation) |
  | log-MAE | **0.064** | 0.065 |
  | pooled MAPE over all held-out flows | **7.02** | 7.21 (GBDT + prior) |
* **Significance** (`loto_significance.txt`, Holm over 54 tests): ours is significantly better
  in **52 of 54** (model × topology) comparisons, with 0 significant losses. The 2 ties are both
  on Germany50: GBDT + prior (+0.05, CI [−0.11, +0.18]) and the readout-only ablation (+0.04).
* **Against the strongest models** (difference, % of archives where ours is better):

  | ours vs | Abilene | Nobel-EU | Germany50 |
  |---|---|---|---|
  | RouteNet-Fermi | −0.87 (98%) | −0.94 (96%) | −0.83 (92%) |
  | QT-RouteNet + transmission term | −2.27 (99%) | −3.58 (100%) | −1.82 (99%) |
  | RouteNet-style + prior features | −0.32 (95%) | −0.14 (82%) | −0.24 (83%) |
  | GBDT + prior | −0.71 (92%) | −0.77 (83%) | +0.05 (n.s.) |
* **Ensemble of 5 seeds** (a different estimand, for reference only):
  - ours: 2.68 / 7.74 / 8.29
  - RouteNet-style + prior features: 2.85 / 7.85 / 8.52
  - RouteNet-Fermi: 3.37 / 8.37 / 8.84

## 4. Where the gain comes from (`results/analysis/prior_ablation.txt`, `figures/fig_prior_ablation.pdf`)
1. **The analytic prior is the main ingredient.** Giving it (plus the packet-size ratio) to any
   learner cuts error a lot:
   - MLP: 9.52 → 7.07
   - GBDT: 9.44 → 6.86
   - GAT: 9.39 → 7.02

   The prior alone is weak (11.09). Learning a correction to it is what works.
2. **Graph learning adds a consistent gain** over tabular learners given the same information
   (6.39 vs 6.86 for GBDT and 7.07 for MLP). The exception is Germany50, where GBDT + prior ties.
3. **Using the prior as the readout base, not only as an input, helps:** 6.39 vs 6.62 with the
   same backbone and inputs. This is significant on all three topologies.
4. **Which prior** (readout base only, same backbone; held-out avg):

   | per-hop prior | Abilene | Nobel-EU | Germany50 | avg |
   |---|---|---|---|---|
   | transmission time only (no queueing prior) | 4.11 | 9.54 | 9.22 | 7.62 |
   | tx/(1 − ρ) (M/M/1 on the flow's packet size) | 3.86 | 9.49 | 9.18 | 7.51 |
   | tx + FIFO M/M/1 waiting | 2.80 | 8.20 | 8.72 | 6.57 |
   | tx + QoS-aware M/M/1/K waiting | 3.02 | 8.34 | 8.63 | 6.66 |
   | **tx + FIFO M/M/1/K waiting (ours)** | 2.83 | **8.15** | **8.46** | **6.48** |

   Decomposing the delay into the flow's own transmission time plus shared waiting gives most of
   the gain; finite buffers add a little (`docs/THEORY.md`).
5. **QT-RouteNet,** the closest earlier queueing-informed GNN, reads delay out as occupancy ×
   buffer / capacity. It has no per-flow transmission term, so it cannot separate flows with
   different packet sizes on the same port (avg 33.25). Adding just that term recovers most of
   the gap (8.94); our full design is best (6.39).

## 5. Budget check (`results/analysis/budget.txt`)
Official RouteNet-Fermi schedule: batch 1, 50 × 2000 steps, Adam 1e-3. Germany50 held out,
seed 0:

| model | held-out | in-distribution (GEANT / Abilene / Nobel-EU) |
|---|---|---|
| RouteNet-Fermi, 75 epochs (standard) | 9.33 | 3.48 / 2.00 / 6.96 |
| RouteNet-Fermi, official schedule | 9.73 | 3.07 / 1.42 / 5.86 |
| QPR-GNN, 25 epochs (standard) | 8.83 | 3.86 / 1.93 / 6.66 |
| **QPR-GNN, official schedule** | **7.97** | 3.16 / **1.23** / **5.20** |

* The longer schedule fits the training topologies better for both models.
* It transfers for ours (8.83 → 7.97) but not for RouteNet-Fermi (9.33 → 9.73).
* The main table uses the cheaper standard protocol for every model, which is conservative
  for us.
* Every learned baseline changes by ≤ 0.16 points of validation MAPE over its last 5 epochs.

## 6. Confirmatory test: 16 new simulated topologies (`results/analysis/sim_folds.txt`, `figures/fig_sim_folds.pdf`)
Design (`docs/SIMULATION.md`):
* 16 Topology Zoo networks (10–45 nodes) simulated with BNNetSimulator, 60 samples each.
* Declared load strata: moderate, near saturation, overload, and fixed capacity with scaled
  traffic.
* 4 pre-declared folds of 4 held-out topologies; every model retrained on the other 12; 3 seeds.
* Method and hyper-parameters were frozen beforehand.

| model | avg over 16 topologies | pooled | overload stratum | ours better on (topologies) |
|---|---|---|---|---|
| M/M/1/K (our prior alone) | 21.78 | 22.98 | 25.56 | 16/16 |
| Gradient-boosted trees | 8.70 | 9.47 | 18.29 | 16/16 |
| GAT | 9.57 | 9.82 | 18.97 | 16/16 |
| RouteNet-style (87k) | 7.12 | 7.44 | 15.89 | 16/16 |
| QT-RouteNet (published design) | 10.28 | 9.75 | 13.15 | 16/16 |
| RouteNet-Fermi | 4.64 | 4.62 | 10.43 | 13/16 |
| QT-RouteNet + transmission term | 4.52 | 4.50 | 8.26 | 13/16 |
| GBDT + prior | **4.14** | 4.28 | 8.36 | 4/16 |
| RouteNet-style + prior features | 4.29 | 4.30 | 8.27 | 7/16 |
| ours, prior as readout base only | 4.55 | 4.59 | 9.18 | 13/16 |
| **QPR-GNN** | 4.27 | **4.27** | 8.49 | — |

**Topology-level paired tests over the 16 topologies:**

| ours vs | mean difference | p (Wilcoxon) |
|---|---|---|
| RouteNet-Fermi | −0.36 (−7.8%) | 0.0013 |
| QT-RouteNet + transmission term | −0.25 (−5.5%) | 0.006 |
| GBDT + prior | +0.13 | 0.21 (n.s.) |
| RouteNet-style + prior features | −0.01 | 0.98 (n.s.) |

**Scenario-level tests:** sign-flip test, Holm over 160 (model, topology) tests. Ours has
92 significant wins and 3 significant losses (all against GBDT + prior).

**Reading:**
* The large gains over models without the prior, and over both published GNN baselines,
  replicate on new topologies.
* Against the two matched-information controls, ours ties on average and has the lowest
  pooled error.

## 6b. Second simulator: the same 16 topologies in ns-3 (`results/analysis/ns3_folds.txt`, `docs/NS3.md`)
* **Design:** fixed before any ns-3 result existed, with the same folds, models, seeds and
  statistics as §6.
* **Data:** ns-3.48, 960 samples simulated, 844 kept. A sample is dropped when some flow
  received no packet: in a dropped sample usually 1 of up to 1,980 flows, a low-rate flow that
  sent nothing, or only a few packets, in the measurement window. Kept per stratum: moderate 90%,
  near saturation 90%, overload 87%, traffic-scaled 83%.
* **Reproducibility:** the 620 samples simulated both on the lab server and here are
  byte-identical. Recomputing the table here from the server's trained models gives the
  server's table exactly.

| model | avg over 16 topologies | pooled | overload stratum | ours better on (topologies) | topology-level p (Wilcoxon) |
|---|---|---|---|---|---|
| M/M/1/K (our prior alone) | 21.07 | 21.91 | 31.29 | 16/16 | < 0.001 |
| Gradient-boosted trees | 14.04 | 13.84 | 28.51 | 16/16 | < 0.001 |
| QT-RouteNet (published design) | 13.85 | 12.51 | 20.00 | 16/16 | < 0.001 |
| GAT | 13.32 | 12.94 | 23.80 | 16/16 | < 0.001 |
| RouteNet-style (87k) | 12.60 | 11.79 | 27.75 | 16/16 | < 0.001 |
| GBDT + prior | 9.39 | 8.33 | 17.34 | 15/16 | < 0.001 |
| ours, prior as readout base only | 9.34 | 8.34 | 17.71 | 15/16 | < 0.001 |
| QT-RouteNet + transmission term | 9.25 | 8.21 | **15.90** | 12/16 | 0.03 |
| RouteNet-Fermi | 9.24 | 8.39 | 18.03 | 10/16 | 0.27 (n.s.) |
| RouteNet-style + prior features | 9.14 | 8.17 | 16.85 | 10/16 | 0.06 (n.s.) |
| **QPR-GNN** | **9.01** | **8.03** | 16.60 | — | — |

**Scenario-level tests:** sign-flip test, Holm over 160 (model, topology) tests. Ours has
**90 significant wins and 0 significant losses**.

**Reading:**
* On an independent simulator ours has the lowest average and pooled error of all 11 models,
  and no significant loss.
* The large gains over models without the prior and over published QT-RouteNet replicate.
* GBDT + prior, which tied with ours on BNNetSimulator, is now clearly worse (15 of 16
  topologies).
* RouteNet-Fermi and RouteNet-style + prior features are close: ours is better on 10 of 16
  topologies, −2.4% and −1.4%, not significant at topology level.
* By stratum, ours is best at near saturation (8.04), and level with the best model at moderate
  load (3.76 vs 3.75) and with fixed capacity (13.66 vs 13.59). Under overload, QT-RouteNet +
  transmission term is best (15.90 vs 16.60).

**Across both simulators (32 held-out topology instances):**

| ours vs | ours better on |
|---|---|
| RouteNet-Fermi | 23/32 |
| QT-RouteNet + transmission term | 24/32 |
| GBDT + prior | 19/32 |
| RouteNet-style + prior features | 17/32 |

Ours has the lowest pooled error on both simulators.

## 7. IoT operating decision (`results/analysis/iot_decision.txt`)
* **Task:** deadline-violation detection. A flow is flagged when its predicted delay exceeds a
  budget of c × its own transmission time.
* **Metric:** F1, macro over the 3 held-out topologies and 5 seeds:

| flows | ours | RouteNet-Fermi | RouteNet-style (87k) | best matched control |
|---|---|---|---|---|
| all, c=2 | 0.745 | 0.649 | 0.704 | 0.748 (RouteNet + prior features) |
| all, c=3 | **0.656** | 0.486 | 0.608 | 0.656 (RouteNet + prior features) |
| all, c=5 | **0.566** | 0.147 | 0.503 | 0.538 (RouteNet + prior features) |
| URLLC, c=5 | **0.203** | 0.047 | 0.141 | 0.194 (RouteNet + prior features) |

* False-alarm rates are ≤ 1.5% for all models.
* Ours catches far more real violations than RouteNet-Fermi (recall 0.59 vs 0.40 at c=3).
* URLLC recall stays low for every model (at most 0.44 at c=2 and 0.13 at c=5), which is a limitation.

## 8. Paper storyline (six claims) and limitations
1. A per-hop decomposition, the flow's own transmission plus shared FIFO M/M/1/K waiting, gives
   a transferable analytic prior (§1, `THEORY.md`, §4.4).
2. Learning a residual on that prior beats 18 models on unseen topologies, including RouteNet-Fermi
   and QT-RouteNet (§3).
3. The gain comes from the prior and from using it as the readout base; graph learning adds a
   further, smaller gain (§4).
4. It holds at RouteNet-Fermi's own training budget (§5).
5. It replicates on 16 new topologies under declared load strata, on two independent simulators
   (BNNetSimulator §6, ns-3 §6b).
6. It improves an IoT operating decision, deadline-violation detection (§7).

Limitations to state:
* The margin over the information-matched controls is modest. Ours ties with GBDT + prior on
  Germany50 and on the BNNetSimulator average (but is better on ns-3, 15/16 topologies). Against
  RouteNet + prior features and RouteNet-Fermi on ns-3, ours is better on 10 of 16 topologies,
  which is not significant at topology level.
* ns-3 uses FIFO and strict priority only, because ns-3 has no weighted fair-queueing queue disc.
  12% of ns-3 samples are dropped by the unmeasurable-flow rule; drops are spread evenly over the
  load strata.
* Transfer holds within a traffic regime. Frozen archive-trained models fail on heavily loaded
  simulated traffic (`results/analysis/regime_shift.txt`).
* URLLC deadline recall is low. Flows with packets much smaller than the port average are
  harder (`THEORY.md`).
* RouteNet-Fermi and QT-RouteNet are PyTorch ports, audited input by input against the official
  code. The remaining difference is that z-score constants are fitted on our training split.
* The FIFO decomposition is exact for FIFO under Poisson-like arrivals. For SP/WFQ/DRR and
  bursty traffic it is a prior that the network corrects, not a theorem.
* The model is an offline what-if tool that needs known offered traffic. Live use would need
  measured rates from an earlier window.
