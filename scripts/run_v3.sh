#!/bin/bash
cd "$(dirname "$0")/.."
P=${P:-2}
jobs=()
for h in test_abilene test_nobel test_germany50; do
  for s in 0 1 2; do
    jobs+=("--name v3_D --readout decoupled --prior_feats --holdout $h --seed $s")
    jobs+=("--name v3_M --readout qosres_nosched --prior_feats --loss mape --holdout $h --seed $s")
    jobs+=("--name v3_DM --readout decoupled --prior_feats --loss mape --holdout $h --seed $s")
  done
done
todo=()
for j in "${jobs[@]}"; do
  n=$(echo "$j" | sed 's/.*--name \([^ ]*\).*/\1/')
  h=$(echo "$j" | sed 's/.*--holdout \([^ ]*\).*/\1/')
  s=$(echo "$j" | sed 's/.*--seed \([^ ]*\).*/\1/')
  [ -f "results/v3/${n}_short_${h}_s${s}.json" ] || todo+=("$j")
done
printf '%s\n' "${todo[@]}" | xargs -P "$P" -I{} sh -c 'python3 -I scripts/train.py {} --out results/v3 > /dev/null 2>&1 && echo "done {}"'
