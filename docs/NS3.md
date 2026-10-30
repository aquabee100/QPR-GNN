# Second simulator: ns-3 (pre-declared design)

**Purpose.** Every labelled dataset so far comes from one simulator family: the archive and the
16 confirmatory topologies are both BNNetSimulator. This experiment repeats the confirmatory test
on the same 16 topologies simulated with **ns-3** (an independent packet-level simulator), so the
claim becomes "transfer to unseen topologies, confirmed on two independent simulators".

**Fixed before any ns-3 result existed (2026-10-22).** Method, hyper-parameters, model list,
folds, seeds and metrics are those of the BNNetSimulator confirmatory test
(`scripts/run_sim_folds.sh`, `scripts/eval_sim_folds.py`); nothing is tuned on ns-3 data.
Whatever the outcome, it is reported.

## Dataset (`scripts/make_ns3_inputs.py`, `ns3/qpr-scenario.cc`; ns-3.48)
| aspect | setting |
|---|---|
| topologies | the same 16 Topology Zoo networks (`results/sim_topologies.csv`), 60 samples each |
| load strata | as before: moderate, near saturation, overload, fixed capacity with traffic scaled ×U(0.5, 2.5) |
| traffic matrix | the same gravity generator; ToS mix 5/5/90%; rates and capacities ×1000 vs BNNetSimulator (0.2–2 Mbit/s per flow; utilisations unchanged) |
| arrivals | Poisson (70% of flows) or ON-OFF (30%; exponential ON/OFF periods with means U(0.2, 1.0) s, Poisson during ON at the rate that keeps the mean) |
| packet sizes | per flow one mix (IP bytes): 300/1700, 500/1000/1400 or 1000 |
| links | one point-to-point link per edge, no propagation delay, MTU 1800 |
| scheduling | per node FIFO (50%) or strict priority with 2 or 3 bands (50%), ToS 0 highest; ns-3 `FifoQueueDisc` / `PrioQueueDisc`; device queue 1 packet |
| buffers | one per node, 8, 16, 32 or 64 KB, applied to every queue |
| routing | static per-destination shortest-path trees: hop count, plus 3 random link-weight perturbations |
| duration | median flow sends about 400 packets (T in [5, 60] s); the first 10% is warm-up |
| label | per-flow mean one-way delay of received packets (FlowMonitor); a scenario where some flow received nothing is dropped (the archive rule) |
| inputs to the models | generated traffic measured at the sources (as in the archive), topology, capacities, scheduling, buffers, routing |

**Differences from BNNetSimulator, by necessity:**
* WFQ and DRR are not used, because ns-3 has no queue disc with configurable weights.
* Buffers are in KB of real IP packets, rather than in the simulator's own units.

**Checks on a 3-topology pilot (2026-10-22):**
* No flow is faster than its pure transmission time, so routes and rates are correct.
* Median measured/prior delay ratio is about 0.9 in every stratum.
* Strict priority favours ToS 0. At high load (fixed-capacity stratum), delay/prior is 0.92 for
  ToS 0 vs 2.37 for ToS 2.
* Reruns are byte-identical. About 40 MB of memory per process.
* Runtime (optimized build): about 1–2 s per sample on Nsfnet and 1–2 min on the largest
  topologies.

## Result (2026-10-28; `results/analysis/ns3_folds.txt`, full table in `docs/RESULTS.md` §6b)
* **Data:**
  - 960 samples simulated on the lab server; 844 kept.
  - A sample is dropped when one of its flows received nothing. Usually that is one low-rate
    flow that sent nothing, or only a few packets, in the window.
  - Kept per stratum: moderate 90%, near saturation 90%, overload 87%, traffic-scaled 83%.
* **Main result:** ours has the lowest average (9.01) and pooled (8.03) MAPE of all 11 models.
  Scenario-level tests: 90 significant wins of 160, 0 losses.
* **Against the close models:** ours is better on 15/16 topologies than GBDT + prior, 12/16
  than QT-RouteNet + transmission term, and 10/16 than RouteNet-Fermi and than RouteNet-style
  + prior features.
* **Reproducibility:** the 620 samples simulated both on the lab server (default build) and
  here (optimized build) are byte-identical. Re-evaluating the server's trained models here
  reproduces the table exactly.
* **Stored:** raw outputs in `data/raw/ns3_results.zip`; trained runs in `results/ns3fold/`.

## Run (lab server; ns-3 already built in `~/software/ns-3-dev`)
```bash
# the lab server has no GitHub access: the repo (code + ns3_inputs) was uploaded as a zip
cd ~/Desktop/project_iot-novem-graphml/ns3-experiments/qpr-gnn
export PATH=~/hodge-venv/bin:$PATH
NS3_DIR=~/software/ns-3-dev P=8 scripts/run_ns3.sh Nsfnet            # pilot (~1 min)
NS3_DIR=~/software/ns-3-dev P=8 nohup scripts/run_ns3.sh > ns3_run.log 2>&1 &   # all 16
```
* `run_ns3.sh` links `ns3/qpr-scenario.cc` into the ns-3 `scratch/` folder and builds it; nothing
  else in the ns-3 tree changes.
* Inputs and outputs stay in this repository: `ns3_inputs/<topology>/results/`.
* The script resumes after an interruption; `ns3_inputs/<topology>/timing.log` has the
  per-sample times.

Then train and evaluate exactly as for BNNetSimulator:
```bash
DATASET=ns3 P=3 NT=4 nohup scripts/run_sim_folds.sh > ns3_folds.log 2>&1 &      # 120 runs
python3 -I scripts/eval_sim_folds.py results/ns3fold                            # -> results/analysis/ns3_folds.txt
```
Bring the raw outputs back with
`cd ns3_inputs && zip -qr ../ns3_results.zip */results */samples.csv */timing.log`.
They are stored as `data/raw/ns3_results.zip`.
