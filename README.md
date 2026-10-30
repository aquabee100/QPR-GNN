# QPR-GNN: queueing-informed graph learning for transferable delay prediction

QPR-GNN predicts mean end-to-end flow delay on network topologies not used for training. It combines each flow's packet transmission time with a port-level waiting-time estimate from a finite-buffer queueing model. A flow–link graph neural network learns a correction at each hop. The queueing estimate is used both as an input and as the base of the delay prediction.

The model and queueing formulation are implemented in `qprgnn/model.py` and `qprgnn/queueing.py`. The derivation is described in [`docs/THEORY.md`](docs/THEORY.md).

## Repository contents

- `qprgnn/` — model, queueing estimates, baseline implementations, and data handling.
- `scripts/` — training, evaluation, and figure-generation scripts.
- `results/` — recorded metrics, run summaries, and statistical analyses.
- `figures/` — figures generated from the experimental results.
- `ns3/` — ns-3 simulation scenario.
- `tests/` — tests for the queueing calculations, model, and dataset parser.
- `docs/` — experimental protocol and simulator notes.

## Setup

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the included tests:

```bash
python -I tests/test_queueing.py
python -I tests/test_model.py
python -I tests/test_parse.py
```

The figures can be regenerated from the included result summaries with:

```bash
python -I scripts/figures.py
```

## Data and experiments

The repository includes experimental summaries, but **not the raw datasets or trained model checkpoints**. The original RouteNet-Fermi trace data and the Topology Zoo graphs are required to prepare the datasets. The BNNetSimulator and ns-3 procedures are documented in [`docs/SIMULATION.md`](docs/SIMULATION.md) and [`docs/NS3.md`](docs/NS3.md).

Once the required datasets have been placed in `data/raw/`, the preparation script can be run from a Bash environment:

```bash
bash scripts/setup_data.sh
```

Training and evaluation scripts are in `scripts/`. The results reported in the manuscript are summarized in [`docs/RESULTS.md`](docs/RESULTS.md), with supporting records in `results/analysis/`.
