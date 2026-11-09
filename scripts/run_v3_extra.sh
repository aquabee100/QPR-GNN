#!/bin/bash
cd "$(dirname "$0")/.."
: ${PHASES:="iot priorabl budget simextra"}
TR="python3 -I scripts/train.py"; BL="python3 -I scripts/train_baselines.py"
jobs=()   # "result json|command"
add() { jobs+=("$1|$2"); }
folds() { python3 -c "import json;d=json.load(open('$1/folds.json'));print(' '.join(d))"; }
for ph in $PHASES; do case $ph in
iot)
  O=results/iotfold; mkdir -p $O
  [ -f $O/folds.json ] || sed 's/ns3_/iot_/g' results/ns3fold/folds.json > $O/folds.json
  ls cache/iot_*.pkl > /dev/null 2>&1 || continue
  U=$(python3 -c "import json;print(','.join(json.load(open('$O/folds.json')).values()))")
  for s in 0 1 2; do for f in f0 f1 f2 f3; do
    T=$(python3 -c "import json;print(json.load(open('$O/folds.json'))['$f'])")
    C="--universe $U --test_topos $T --holdout $f --seed $s --out $O"; X="${f}_s${s}"
    add $O/routenet_qosres_nosched_pf_short_$X.json "$TR --name routenet_qosres_nosched_pf --readout qosres_nosched --prior_feats --loss mape $C"
    add $O/routenet_txres_pf_short_$X.json "$TR --name routenet_txres_pf --readout tx_res --prior_feats --loss mape $C"
    add $O/routenet_qosres_nosched_short_$X.json "$TR --name routenet_qosres_nosched --readout qosres_nosched --loss mape $C"
    for m in gbdt_p gbdt gat mlp gcn gtrans; do add $O/bl_${m}_short_$X.json "$BL --model $m $C"; done
    add $O/routenet_short_$X.json "$TR --name routenet --readout mm1res $C"
    add $O/routenet_h60_short_$X.json "$TR --name routenet_h60 --readout mm1res --H 60 $C"
    add $O/qtroutenet_short_$X.json "python3 -I scripts/train_qtroutenet.py $C"
    add $O/qtroutenet_tx_short_$X.json "python3 -I scripts/train_qtroutenet.py --tx_term $C"
    add $O/fermi_official_short_$X.json "python3 -I scripts/train_fermi.py --epochs 75 $C"
  done; done ;;
priorabl)
  for s in 0 1 2 3 4; do for h in test_abilene test_nobel test_germany50; do
    for spec in "routenet_tx_res tx_res" "routenet mm1res" "routenet_qosres_mm1 qosres_mm1" "routenet_qosres qosres"; do
      set -- $spec
      add results/loto_mape/$1_short_${h}_s$s.json "$TR --name $1 --readout $2 --loss mape --holdout $h --seed $s --out results/loto_mape"
    done
  done; done ;;
budget)
  add results/budget/routenet_qosres_nosched_pf_official_mape_short_test_germany50_s0.json \
      "$TR --name routenet_qosres_nosched_pf_official_mape --readout qosres_nosched --prior_feats --loss mape --holdout test_germany50 --seed 0 --bs 1 --epochs 50 --steps_per_epoch 2000 --lr 1e-3 --out results/budget" ;;
msmp)
  [ "${MSMP_ARCHIVE:-1}" = 1 ] && for s in 0 1 2 3 4; do for h in test_abilene test_nobel test_germany50; do
    add results/loto/msmp_attn_short_${h}_s$s.json "python3 -I scripts/train_msmp.py --holdout $h --seed $s --out results/loto"
  done; done
  for D in simfold ns3fold iotfold; do
    [ "$D" = iotfold ] && ! ls cache/iot_*.pkl > /dev/null 2>&1 && continue
    U=$(python3 -c "import json;print(','.join(json.load(open('results/$D/folds.json')).values()))")
    for s in 0 1 2; do for f in f0 f1 f2 f3; do
      T=$(python3 -c "import json;print(json.load(open('results/$D/folds.json'))['$f'])")
      add results/$D/msmp_attn_short_${f}_s$s.json "python3 -I scripts/train_msmp.py --universe $U --test_topos $T --holdout $f --seed $s --out results/$D"
    done; done
  done ;;
fill)
  for D in simfold ns3fold iotfold; do
    [ "$D" = iotfold ] && ! ls cache/iot_*.pkl > /dev/null 2>&1 && continue
    O=results/${D}_mape; [ "$D" = iotfold ] && O=results/iotfold; mkdir -p $O
    U=$(python3 -c "import json;print(','.join(json.load(open('results/$D/folds.json')).values()))")
    for s in 0 1 2; do for f in f0 f1 f2 f3; do
      T=$(python3 -c "import json;print(json.load(open('results/$D/folds.json'))['$f'])")
      for m in mlp_p gat_p; do
        add $O/bl_${m}_short_${f}_s$s.json "$BL --model $m --loss mape --universe $U --test_topos $T --holdout $f --seed $s --out $O"
      done
      if [ "$D" = iotfold ]; then
        mkdir -p results/iotfold_log
        add results/iotfold_log/routenet_qosres_nosched_pf_short_${f}_s$s.json "$TR --name routenet_qosres_nosched_pf --readout qosres_nosched --prior_feats --universe $U --test_topos $T --holdout $f --seed $s --out results/iotfold_log"
      fi
    done; done
  done ;;
simextra)
  for D in simfold ns3fold; do
    U=$(python3 -c "import json;print(','.join(json.load(open('results/$D/folds.json')).values()))")
    for s in 0 1 2; do for f in f0 f1 f2 f3; do
      T=$(python3 -c "import json;print(json.load(open('results/$D/folds.json'))['$f'])")
      C="--universe $U --test_topos $T --holdout $f --seed $s --out results/$D"; X="${f}_s${s}"
      for m in mlp gcn gtrans; do add results/$D/bl_${m}_short_$X.json "$BL --model $m $C"; done
      add results/$D/routenet_short_$X.json "$TR --name routenet --readout mm1res $C"
    done; done
  done ;;
esac; done
todo=()
for j in "${jobs[@]}"; do
  out=${j%%|*}; cmd=${j#*|}
  [ -f "$out" ] && continue
  pgrep -f -- "$cmd" > /dev/null && continue
  todo+=("$cmd")
done
echo "${#todo[@]} runs to do"
[ -n "$DRY" ] && { printf '%s\n' "${todo[@]}"; exit 0; }
[ ${#todo[@]} -eq 0 ] && exit 0
if [ -n "$ONE" ]; then NTHREADS=1 sh -c "${todo[0]}" > /dev/null 2>&1 && echo "done ${todo[0]}"; exit 0; fi
printf '%s\n' "${todo[@]}" | xargs -P "${P:-2}" -I{} sh -c 'NTHREADS=1 {} > /dev/null 2>&1 && echo "done {}"'
