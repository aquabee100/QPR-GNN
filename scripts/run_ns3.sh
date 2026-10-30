#!/bin/bash
# Build the scenario, run all samples
# usage: NS3_DIR=<ns-3 tree> P=8 scripts/run_ns3.sh
cd "$(dirname "$0")/.."
: ${NS3_DIR:=$HOME/software/ns-3-dev} ${P:=8} ${IN:=ns3_inputs}
REPO=$(pwd)
[ -d "$IN" ] || python3 -I scripts/make_ns3_inputs.py --out "$IN" || exit 1
ln -sfn "$REPO/ns3/qpr-scenario.cc" "$NS3_DIR/scratch/qpr-scenario.cc"
(cd "$NS3_DIR" && ./ns3 build scratch/qpr-scenario) > "$IN/build.log" 2>&1 || { echo "build failed, see $IN/build.log"; exit 1; }
BIN=$(ls "$NS3_DIR"/build/scratch/ns3*-qpr-scenario-* 2>/dev/null | head -1)
[ -x "$BIN" ] || { echo "binary not found under $NS3_DIR/build/scratch"; exit 1; }
export BIN LD_LIBRARY_PATH="$NS3_DIR/build/lib:$LD_LIBRARY_PATH"
topos="$@"; [ -z "$topos" ] && topos=$(ls -d $IN/*/ | xargs -n1 basename)
for t in $topos; do mkdir -p "$IN/$t/results"; done
one() { f=$1; t=$(basename $(dirname $f)); i=$(basename $f .txt); o=$(dirname $f)/results/$i.out
  [ -s "$o" ] && return
  s=$(date +%s.%N); "$BIN" --in="$f" --out="$o.tmp" > "$o.log" 2>&1 && mv "$o.tmp" "$o" && rm -f "$o.log"
  echo "$i $(awk -v a=$s -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b-a}') $([ -s "$o" ] && echo ok || echo FAIL)" >> $(dirname $f)/timing.log; }
export -f one
for t in $topos; do ls $IN/$t/sample_*.txt; done | xargs -P $P -I{} bash -c 'one {}'
for t in $topos; do echo "$t: $(ls $IN/$t/results/*.out 2>/dev/null | wc -l) / $(ls $IN/$t/sample_*.txt | wc -l) samples"; done
echo "NS3 DONE"
