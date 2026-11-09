#!/bin/bash
cd "$(dirname "$0")/.."
: ${P:=2} ${PHASE:=archive}
jobs=()
if [ "$PHASE" = archive ]; then
  for s in 0 1 2 3 4; do for h in test_abilene test_nobel test_germany50; do
    [ $s -ge 3 ] && jobs+=("routenet_qosres_nosched_pf|results/loto_mape|python3 -I scripts/train.py --name routenet_qosres_nosched_pf --readout qosres_nosched --prior_feats --loss mape --holdout $h --seed $s --out results/loto_mape")
    jobs+=("routenet_txres_pf|results/loto_mape|python3 -I scripts/train.py --name routenet_txres_pf --readout tx_res --prior_feats --loss mape --holdout $h --seed $s --out results/loto_mape")
    for m in mlp_p gat_p; do
      jobs+=("bl_$m|results/loto_mape|python3 -I scripts/train_baselines.py --model $m --loss mape --holdout $h --seed $s --out results/loto_mape"); done
  done; done
elif [ "$PHASE" = ablation ]; then
  for s in 0 1 2 3 4; do for h in test_abilene test_nobel test_germany50; do
    jobs+=("routenet_qosres_nosched|results/loto_mape|python3 -I scripts/train.py --name routenet_qosres_nosched --readout qosres_nosched --loss mape --holdout $h --seed $s --out results/loto_mape")
  done; done
  for D in simfold ns3fold; do
    U=$(python3 -c "import json;print(','.join(json.load(open('results/$D/folds.json')).values()))")
    for s in 0 1 2; do for f in f0 f1 f2 f3; do
      T=$(python3 -c "import json;print(json.load(open('results/$D/folds.json'))['$f'])")
      jobs+=("routenet_qosres_nosched|results/${D}_mape|python3 -I scripts/train.py --name routenet_qosres_nosched --readout qosres_nosched --loss mape --universe $U --test_topos $T --holdout $f --seed $s --out results/${D}_mape")
    done; done
  done
else
  for D in simfold ns3fold; do
    U=$(python3 -c "import json;print(','.join(json.load(open('results/$D/folds.json')).values()))")
    for s in 0 1 2; do for f in f0 f1 f2 f3; do
      T=$(python3 -c "import json;print(json.load(open('results/$D/folds.json'))['$f'])")
      C="--universe $U --test_topos $T --holdout $f --seed $s --out results/${D}_mape"
      jobs+=("routenet_qosres_nosched_pf|results/${D}_mape|python3 -I scripts/train.py --name routenet_qosres_nosched_pf --readout qosres_nosched --prior_feats --loss mape $C")
      [ -n "$CTL" ] && jobs+=("routenet_txres_pf|results/${D}_mape|python3 -I scripts/train.py --name routenet_txres_pf --readout tx_res --prior_feats --loss mape $C")
    done; done
  done
fi
todo=()
for j in "${jobs[@]}"; do
  IFS='|' read -r n o cmd <<< "$j"
  h=$(echo "$cmd" | sed 's/.*--holdout \([^ ]*\).*/\1/'); s=$(echo "$cmd" | sed 's/.*--seed \([^ ]*\).*/\1/')
  [ -f "$o/${n}_short_${h}_s${s}.json" ] && continue
  pgrep -f -- "$cmd" > /dev/null && continue                    # already running
  todo+=("$cmd")
done
echo "${#todo[@]} runs to do"
[ -n "$DRY" ] && { printf '%s\n' "${todo[@]}"; exit 0; }
[ -n "$ONE" ] && { NTHREADS=1 sh -c "${todo[0]}" > /dev/null 2>&1 && echo "done ${todo[0]}"; exit 0; }
printf '%s\n' "${todo[@]}" | xargs -P "$P" -I{} sh -c 'NTHREADS=1 {} > /dev/null 2>&1 && echo "done {}"'
echo "V3 EVAL $PHASE DONE"
