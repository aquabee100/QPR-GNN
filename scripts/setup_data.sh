#!/bin/bash
# Extract raw data and build caches
set -e
cd "$(dirname "$0")/.."
mkdir -p data/zoo data/traces cache
[ -d data/zoo/graphml ] || tar -xzf data/raw/topology_zoo_graphml.tar.gz -C data/zoo
[ -d data/traces/real_traces ] || (cd data/traces && unzip -q ../raw/real_traces.zip)
ln -sfn data/zoo/graphml data_zoo
ln -sfn data/traces/real_traces data_traces
python3 -I scripts/build_cache.py train/geant test/nobel test/germany50 test/abilene
# BNNetSimulator topologies: inputs and outputs
if [ ! -d sim_inputs/Nsfnet/results ]; then
  python3 -I scripts/make_sim_inputs.py --out sim_inputs > /dev/null
  unzip -qo data/raw/sim_results.zip
  python3 -I scripts/prepare_sim.py sim_inputs
fi
# ns-3 topologies: inputs and outputs
if [ ! -d ns3_inputs/Nsfnet/results ]; then
  python3 -I scripts/make_ns3_inputs.py --out ns3_inputs > /dev/null
  unzip -qo data/raw/ns3_results.zip -d ns3_inputs
  python3 -I scripts/prepare_ns3.py ns3_inputs
fi
echo "data ready"
