#!/bin/bash
# Official RouteNet-Fermi schedule, Germany50, seed 0
cd "$(dirname "$0")/.."
mkdir -p results/budget logs
C="--holdout test_germany50 --seed 0 --bs 1 --epochs 50 --steps_per_epoch 2000 --lr 1e-3 --out results/budget"
NTHREADS=1 python3 -I scripts/train_fermi.py $C > logs/budget_fermi.txt 2>&1 &
NTHREADS=1 python3 -I scripts/train.py --name routenet_qosres_nosched_pf_official --readout qosres_nosched --prior_feats $C \
  > logs/budget_ours.txt 2>&1 &
wait; echo "BUDGET DONE"
