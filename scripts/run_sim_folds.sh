#!/bin/bash
# Cross-topology folds on 16 simulated topologies
# usage: P=3 NT=4 [DATASET=ns3] scripts/run_sim_folds.sh
cd "$(dirname "$0")/.."
: ${DATASET:=sim} ${P:=3} ${SEEDS:="0 1 2"}
# DATASET=sim (BNNetSimulator) or ns3
if [ "$DATASET" = ns3 ]; then : ${OUT:=results/ns3fold}; PREP="python3 -I scripts/prepare_ns3.py ns3_inputs"
else : ${OUT:=results/simfold}; PREP="python3 -I scripts/prepare_sim.py sim_inputs"; fi
mkdir -p $OUT
D=$DATASET
F0=${D}_Ai3,${D}_Xspedius,${D}_Geant,${D}_Palmetto
F1=${D}_Abilene,${D}_Integra,${D}_Intranetwork,${D}_Carnet
F2=${D}_HiberniaCanada,${D}_Ibm,${D}_SwitchL3,${D}_Sanet
F3=${D}_Nsfnet,${D}_Ans,${D}_LambdaNet,${D}_Litnet
U=$F0,$F1,$F2,$F3
cat > $OUT/folds.json <<EOF
{"f0": "$F0", "f1": "$F1", "f2": "$F2", "f3": "$F3"}
EOF
$PREP || exit 1
export U OUT
jobs=()
for s in $SEEDS; do for f in f0 f1 f2 f3; do
  for m in fermi2 ours prev ctl gbdt_p routenet_h60 gbdt gat qt qttx; do jobs+=("$m $f $s"); done; done; done
run() { set -- $1; m=$1 f=$2 s=$3
  T=$(python3 -c "import json;print(json.load(open('$OUT/folds.json'))['$f'])")
  C="--universe $U --test_topos $T --holdout $f --seed $s --out $OUT"
  case $m in
    ours)  [ -f $OUT/routenet_qosres_nosched_pf_short_${f}_s${s}.json ] && return
           NTHREADS=${NT:-1} python3 -I scripts/train.py --name routenet_qosres_nosched_pf --readout qosres_nosched --prior_feats $C ;;
    prev)  [ -f $OUT/routenet_qosres_nosched_short_${f}_s${s}.json ] && return
           NTHREADS=${NT:-1} python3 -I scripts/train.py --name routenet_qosres_nosched --readout qosres_nosched $C ;;
    ctl)   [ -f $OUT/routenet_txres_pf_short_${f}_s${s}.json ] && return
           NTHREADS=${NT:-1} python3 -I scripts/train.py --name routenet_txres_pf --readout tx_res --prior_feats $C ;;
    routenet_h60) [ -f $OUT/routenet_h60_short_${f}_s${s}.json ] && return
           NTHREADS=${NT:-1} python3 -I scripts/train.py --name routenet_h60 --readout mm1res --H 60 $C ;;
    fermi2) [ -f $OUT/fermi_official_short_${f}_s${s}.json ] && return
           NTHREADS=${NT:-1} python3 -I scripts/train_fermi.py --epochs 75 $C ;;
    qt)    [ -f $OUT/qtroutenet_short_${f}_s${s}.json ] && return
           NTHREADS=${NT:-1} python3 -I scripts/train_qtroutenet.py $C ;;
    qttx)  [ -f $OUT/qtroutenet_tx_short_${f}_s${s}.json ] && return
           NTHREADS=${NT:-1} python3 -I scripts/train_qtroutenet.py --tx_term $C ;;
    gbdt|gbdt_p|gat) [ -f $OUT/bl_${m}_short_${f}_s${s}.json ] && return
           NTHREADS=${NT:-1} python3 -I scripts/train_baselines.py --model $m $C ;;
  esac > $OUT/log_${m}_${f}_s${s}.txt 2>&1; }
export -f run
printf '%s\n' "${jobs[@]}" | xargs -P $P -I{} bash -c 'run "{}"'
echo SIM GRID DONE
