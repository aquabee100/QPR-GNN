#!/bin/bash
# All archive leave-one-topology-out runs
# usage: P=4 scripts/run_loto.sh
cd "$(dirname "$0")/.."
: ${P:=4} ${SEEDS:="0 1 2 3 4"} ${OUT:=results/loto}
mkdir -p $OUT logs
TRAIN="python3 -I scripts/train.py"
# name | command
MODELS=(
  "routenet_qosres_nosched_pf|$TRAIN --name routenet_qosres_nosched_pf --readout qosres_nosched --prior_feats"  # OURS
  "routenet_qosres_nosched|$TRAIN --name routenet_qosres_nosched --readout qosres_nosched"     # ablation: prior as readout only
  "routenet_txres_pf|$TRAIN --name routenet_txres_pf --readout tx_res --prior_feats"           # matched information
  "routenet_qosres|$TRAIN --name routenet_qosres --readout qosres"                             # prior ablation
  "routenet_qosres_mm1|$TRAIN --name routenet_qosres_mm1 --readout qosres_mm1"                 # prior ablation
  "routenet_tx_res|$TRAIN --name routenet_tx_res --readout tx_res"                             # prior ablation
  "routenet_fermi|$TRAIN --name routenet_fermi --readout fermi"                                # selection candidate
  "routenet|$TRAIN --name routenet --readout mm1res"                                           # RouteNet-style 25k
  "routenet_h60|$TRAIN --name routenet_h60 --readout mm1res --H 60"                            # RouteNet-style 87k
  "fermi_official|python3 -I scripts/train_fermi.py --epochs 75"                               # RouteNet-Fermi
  "qtroutenet|python3 -I scripts/train_qtroutenet.py"                                          # QT-RouteNet
  "qtroutenet_tx|python3 -I scripts/train_qtroutenet.py --tx_term"
)
for m in mlp gbdt gcn gat gtrans mlp_p gbdt_p gat_p; do
  MODELS+=("bl_$m|python3 -I scripts/train_baselines.py --model $m"); done
jobs=()
for s in $SEEDS; do for h in test_abilene test_nobel test_germany50; do for i in "${!MODELS[@]}"; do
  jobs+=("$i $h $s"); done; done; done
run() { set -- $1; spec=${MODELS[$1]}; name=${spec%%|*}; cmd=${spec#*|}
  [ -f $OUT/${name}_short_$2_s$3.json ] && return
  [ -n "$DRY" ] && { echo "$cmd --holdout $2 --seed $3 --out $OUT"; return; }
  NTHREADS=1 $cmd --holdout $2 --seed $3 --out $OUT > logs/${name}_$2_s$3.txt 2>&1; }
export -f run; export OUT DRY; export MODELS_STR="$(printf '%s\n' "${MODELS[@]}")"
printf '%s\n' "${jobs[@]}" | xargs -P $P -I{} bash -c 'mapfile -t MODELS <<< "$MODELS_STR"; run "{}"'
echo "LOTO GRID DONE"
