#!/bin/bash
# All runs of the benchmark (bench.py), in order of priority. Toto runs in its own environment.
# Set PY and PY_TOTO to the Python interpreters of the two environments, for example
#   PY=.venv/bin/python PY_TOTO=.venv-toto/bin/python bash run_benchmark.sh
set -e
cd "$(dirname "$0")"
PY="${PY:-python}"
PY_TOTO="${PY_TOTO:-python}"

$PY bench.py --data heldout --horizon 24 --models naive bolt bolt_base chronos2 timesfm sundial aurora
$PY bench.py --data aemo    --horizon 24 --models naive bolt bolt_base chronos2 timesfm sundial aurora
$PY_TOTO bench.py --data heldout --horizon 24 --models toto --tag toto
$PY_TOTO bench.py --data aemo    --horizon 24 --models toto --tag toto
for H in 192 720; do
  $PY bench.py --data heldout --horizon $H --n_test 150 --seeds 0 --models naive bolt chronos2 timesfm sundial aurora
  $PY_TOTO bench.py --data heldout --horizon $H --n_test 150 --seeds 0 --models toto --tag toto
done
$PY bench.py --data heldout --horizon 24 --models chronos
$PY bench.py --data aemo    --horizon 24 --seeds 0 --models chronos
$PY bench.py --data heldout --horizon 24 --seeds 0 --models chronos_base --tag base
$PY bench.py --data heldout --horizon 192 --n_test 150 --seeds 0 --models chronos
echo "benchmark done"
