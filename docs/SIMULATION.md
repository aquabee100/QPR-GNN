# Confirmatory test on 16 new simulated topologies

The archive has only three held-out targets, and they were looked at during development. The
confirmatory test therefore uses 16 new topologies simulated with BNNetSimulator (the generator
of the RouteNet-Fermi dataset; https://github.com/BNN-UPC/BNNetSimulator). The method, its
hyper-parameters and every baseline were frozen before these results existed.

## 1. Generate the inputs (deterministic, seed 0)
```bash
scripts/setup_data.sh                                  # extracts the Topology Zoo (data_zoo/)
python3 -I scripts/make_sim_inputs.py --out sim_inputs # 16 topologies x 60 samples
python3 -I scripts/check_sim_inputs.py                 # the simulator's own input checks, re-implemented
python3 -I scripts/audit_sim_inputs.py                 # -> results/analysis/sim_inputs_audit.txt
```
* **Topologies:** Topology Zoo networks with 10–45 nodes, two per bin of the cycle count
  `b1_short` (`results/zoo_graphs.csv`), spread in size: Ai3, Xspedius, Geant, Palmetto, Abilene,
  Integra, Intranetwork, Carnet, HiberniaCanada, Ibm, SwitchL3, Sanet, Nsfnet, Ans, LambdaNet,
  Litnet (`results/sim_topologies.csv`).
* **Per sample:** gravity-model traffic matrix (log-normal node masses), ToS mix 5/5/90%,
  70% Poisson and 30% ON-OFF arrivals, random packet-size mixes, per-node FIFO/SP/WFQ/DRR with
  2–3 queues and one buffer size per node, 4 routing variants per topology.
* **Declared load strata** (recorded in `<topology>/samples.csv`):
  - 75% of samples get capacities sized per sample: *moderate* (max utilisation U(0.1, 0.6)),
    *near_saturation* (most loaded 10% of links at U(0.7, 0.95)), *overload* (U(1.0, 1.2), so
    finite buffers drop packets);
  - 25% form a fixed-capacity family (*traffic_scaled*): one capacity set per topology, traffic
    scaled ×U(0.5, 2.5). This separates traffic shift from capacity sizing.

The simulator outputs used in the paper are in `data/raw/sim_results.zip` (16 topologies,
60 samples each); `scripts/setup_data.sh` regenerates the inputs and unpacks them, so steps 2
and the simulator run can be skipped to reproduce the paper.

## 2. Run the simulator
With Docker:
```bash
cd sim_inputs && ./run_all.sh            # resumes; a topology is skipped only when complete
```
Without Docker or root (shared JupyterHub-style server; tested 2026-10-15; Apptainer cannot
mount there, so the image is unpacked and run with proot):
```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
~/bin/micromamba create -y -p ~/apptainer-env -c conda-forge apptainer squashfs-tools
export PATH=~/apptainer-env/bin:$PATH
apptainer pull bnnetsim.sif docker://bnnupc/bnnetsimulator
apptainer sif list bnnetsim.sif                       # the FS (Squashfs) partition, ID 4 here
apptainer sif dump 4 bnnetsim.sif > rootfs.squashfs && unsquashfs -d bnnetsim_root rootfs.squashfs
curl -Ls -o ~/bin/proot https://proot.gitlab.io/proot/bin/proot && chmod +x ~/bin/proot
cd sim_inputs && THREADS=8 nohup ./run_proot.sh > run.log 2>&1 &
```
Choose THREADS for the memory limit (16 GB there: 8 threads). Each topology writes
`sim_inputs/<topology>/results/<topology>/*.tar.gz`; `status.log` has one line per topology.

## 3. Train and evaluate (4 pre-declared folds of 4 held-out topologies, 3 seeds)
```bash
python3 -m venv ~/qpr-venv && ~/qpr-venv/bin/pip install -r requirements.txt   # python -I ignores --user installs
export PATH=~/qpr-venv/bin:$PATH
P=3 NT=4 scripts/run_sim_folds.sh        # prepare_sim.py, then 11 models x 4 folds x 3 seeds
python3 -I scripts/eval_sim_folds.py     # -> results/analysis/sim_folds.txt, results/simfold/{metrics,tests}.csv
```
`run_sim_folds.sh` first runs `scripts/prepare_sim.py`, which parses every topology with at
least 90% of its samples (the simulator drops samples it marks unstable) into `cache/sim_<T>.pkl`
and records each sample's stratum in `cache/sim_strata.json`. P × NT should not exceed the
cores you may use; on a 16 GB limit, P=3 NT=4 ran without memory errors (P=8 did not).

## 4. Regime shift (limitation)
`scripts/eval_regime_shift.py sim_inputs/Nsfnet` evaluates the archive-trained models,
without retraining, on a simulated topology. All learned models degrade strongly there (the
simulated traffic is heavier and differently shaped than the archive's); see
`results/analysis/regime_shift.txt`. The paper claims transfer across topologies within a
traffic regime, and states this as a limitation.

## Notes
* `qprgnn/parse.py` reads both the archive format and the simulator format; no code shipped
  with the data is executed.
* BNNetSimulator requires one integer `bufferSizes` per node and capacities that are multiples
  of 1000 (≥ 10,000); `make_sim_inputs.py` follows both rules.
