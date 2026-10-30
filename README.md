# QPR-GNN: queueing-informed graph learning for transferable delay prediction

Code, trained runs and results for the paper *"Queueing-Informed Graph Learning for Transferable
Delay Prediction in IoT Backhaul Networks"* (IEEE IoT Journal, special issue on Graph
Representation Learning for IoT; in preparation).

**Task.** Predict the mean end-to-end delay of every flow in a backhaul network from its
topology, routing, scheduling configuration and offered traffic, on topologies never seen in
training.

**Method.** QPR-GNN is a RouteNet-style flow ↔ link message-passing network whose per-hop output
is a learned correction to an analytic queueing prior:

  per-hop delay = (s_f / C + W) · exp(r),

* s_f / C is the flow's own packet transmission time.
* W is the FIFO M/M/1/K waiting time of the port, computed from the aggregate traffic (mean
  packet size s̄, buffer B/s̄ packets; finite also for ρ ≥ 1).
* r is the network's output; its last layer is zero-initialised, so training starts exactly at
  the analytic prior.
* The prior also enters every hop as input features: log(1 + W/tx) and log(s_f/s̄).

Why the decomposition matters: `docs/THEORY.md`.

## Main results
Every number comes from a file in `results/`. Full tables and tests: `docs/RESULTS.md`.

**Unseen archive topologies** (RouteNet-Fermi real-traffic dataset; train on GEANT + 2 others,
test on the third). MAPE %, single model, mean over 5 seeds:

| model | Abilene | Nobel-EU | Germany50 | avg |
|---|---|---|---|---|
| M/M/1/K (our prior alone) | 6.66 | 16.49 | 10.11 | 11.09 |
| Gradient-boosted trees | 5.10 | 11.56 | 11.65 | 9.44 |
| GAT | 5.11 | 11.94 | 11.12 | 9.39 |
| RouteNet-style (87k) | 3.94 | 9.22 | 9.00 | 7.39 |
| RouteNet-Fermi | 3.64 | 8.83 | 9.32 | 7.27 |
| QT-RouteNet + transmission term | 5.04 | 11.47 | 10.32 | 8.94 |
| GBDT + prior (matched information) | 3.49 | 8.66 | 8.45 | 6.86 |
| RouteNet-style + prior features (matched information) | 3.09 | 8.04 | 8.73 | 6.62 |
| **QPR-GNN (ours)** | **2.78** | **7.89** | 8.49 | **6.39** |

**16 new topologies, two independent simulators** (confirmatory; 4 pre-declared folds of 4
held-out topologies, 3 seeds; method frozen beforehand). MAPE % averaged over the 16 held-out
topologies (pooled over all flows in brackets):

| model | BNNetSimulator | ns-3 |
|---|---|---|
| RouteNet-style (87k) | 7.12 (7.44) | 12.60 (11.79) |
| QT-RouteNet (published design) | 10.28 (9.75) | 13.85 (12.51) |
| RouteNet-Fermi | 4.64 (4.62) | 9.24 (8.39) |
| QT-RouteNet + transmission term | 4.52 (4.50) | 9.25 (8.21) |
| GBDT + prior | **4.14** (4.28) | 9.39 (8.33) |
| RouteNet-style + prior features | 4.29 (4.30) | 9.14 (8.17) |
| **QPR-GNN (ours)** | 4.27 (**4.27**) | **9.01** (**8.03**) |

Scenario-level significance (Holm over 160 tests per simulator): 92 wins / 3 losses on
BNNetSimulator, 90 wins / 0 losses on ns-3.

## Repository
```
qprgnn/                     the package
  parse.py                  reader for the RouteNet / BNNetSimulator dataset format
  queueing.py               analytic priors (FIFO M/M/1/K; M/M/1, M/G/1 baselines)
  data.py                   per-sample tensors, batching, archive-level splits, caching
  model.py                  QPRGNN (ours) and the RouteNet-style models sharing its backbone
  baselines.py              MLP, gradient-boosted trees, GCN, GAT, GPS-style transformer (+ prior variants)
  fermi.py                  PyTorch port of RouteNet-Fermi (inputs audited against the official code)
  qtroutenet.py             PyTorch port of QT-RouteNet
  protocol.py               training/evaluation protocol shared by every learned model
  runs.py                   load trained runs, predict on any topology
scripts/                    training, grids, evaluation, simulation, figures (see table below)
tests/                      queueing formulas, model invariances, parser
results/loto/               all archive runs (19 models x 3 held-out topologies x 5 seeds)
results/simfold/            BNNetSimulator-fold runs (120)
results/ns3fold/            ns-3-fold runs (120)
results/budget/             equal-budget runs
results/analysis/           text summaries of every result
figures/                    paper figures
ns3/                        ns-3 scenario (second simulator; docs/NS3.md)
docs/                       RESULTS.md (all numbers), THEORY.md, SIMULATION.md, NS3.md, CFP.md
```

## Claim → script → result
| claim | run | evaluate | result file |
|---|---|---|---|
| Ours beats 18 models on unseen archive topologies | `scripts/run_loto.sh` | `scripts/eval_loto.py` | `results/analysis/loto_table.txt`, `loto_significance.txt` |
| The decomposed prior is the right base | `scripts/run_loto.sh` | `scripts/eval_prior_ablation.py` | `results/analysis/prior_ablation.txt` |
| The design was chosen on validation data only | `scripts/run_loto.sh` | `scripts/selection_check.py` | `results/analysis/selection.txt` |
| Better deadline-violation detection (IoT decision) | — | `scripts/iot_decision.py` | `results/analysis/iot_decision.txt` |
| The gap holds at RouteNet-Fermi's own training budget | `scripts/run_budget.sh` | `scripts/budget_summary.py` | `results/analysis/budget.txt` |
| Confirmed on 16 new simulated topologies | `scripts/make_sim_inputs.py`, simulator, `scripts/run_sim_folds.sh` | `scripts/eval_sim_folds.py` | `results/analysis/sim_folds.txt` |
| ... and again with a second simulator, ns-3 | `scripts/make_ns3_inputs.py`, `scripts/run_ns3.sh`, `DATASET=ns3 scripts/run_sim_folds.sh` | `scripts/eval_sim_folds.py results/ns3fold` | `results/analysis/ns3_folds.txt` |
| Simulator inputs are valid | — | `scripts/check_sim_inputs.py`, `scripts/audit_sim_inputs.py` | `results/analysis/sim_inputs_audit.txt` |
| Data attrition is documented | — | `scripts/attrition.py` | `results/analysis/attrition.txt` |
| Limitation: regime shift archive → loaded simulation | — | `scripts/eval_regime_shift.py sim_inputs/Nsfnet` | `results/analysis/regime_shift.txt` |
| Figures | — | `scripts/figures.py` | `figures/` |

## Reproduce
Raw data goes in `data/raw/` (separate data archive).
Trained weights (.pt, .pkl) are not included; retrain with the run scripts.

```bash
pip install -r requirements.txt
scripts/setup_data.sh                      # extract data/raw/, parse into cache/ (~2 min)
python3 -I tests/test_queueing.py && python3 -I tests/test_model.py && python3 -I tests/test_parse.py

# evaluate the stored runs (needs the trained weights)
python3 -I scripts/eval_loto.py && python3 -I scripts/eval_prior_ablation.py
python3 -I scripts/selection_check.py && python3 -I scripts/iot_decision.py && python3 -I scripts/budget_summary.py
python3 -I scripts/figures.py

# retrain one model (about 25 min on one CPU thread)
python3 -I scripts/train.py --name routenet_qosres_nosched_pf --readout qosres_nosched --prior_feats \
    --holdout test_germany50 --seed 0 --out results/mine
# retrain everything (skips runs that exist; DRY=1 prints the commands)
P=4 scripts/run_loto.sh
```
The simulator experiment (inputs, running BNNetSimulator with or without Docker, training on
the folds) is described in `docs/SIMULATION.md`; the second-simulator replication with ns-3
in `docs/NS3.md`.

## Data
* **RouteNet-Fermi real-traffic dataset** (`data/raw/real_traces.zip`): GEANT, Abilene, Nobel-EU,
  Germany50, packet-level simulations by the BNN-UPC group. The bundled `datanetAPI.py` is not
  used; `qprgnn/parse.py` reads the documented format. GEANT's `train/` and `test/` archives are
  byte-identical, so GEANT is always a training topology with our own archive-level split.
* **Internet Topology Zoo** (`data/raw/topology_zoo_graphml.tar.gz`): source of the 16 simulated
  topologies.
