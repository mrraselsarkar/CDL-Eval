#!/bin/bash
# Detached CPU jobs for the training-budget experiment (budget.py). The GPU is hidden so that these jobs
# do not compete with GPU work. Set PY to the Python interpreter of the environment (default: python).
# Usage: bash run_cpu_jobs.sh budget_bimodal | budget_lorenz
export CUDA_VISIBLE_DEVICES=-1
cd "$(dirname "$0")" || exit 1
PY="${PY:-python}"
mkdir -p logs
case "$1" in
  budget_bimodal) "$PY" -u budget.py --datasets bimodal --threads 4 > logs/budget_bimodal.log 2>&1 ;;
  budget_lorenz)  "$PY" -u budget.py --datasets lorenz --threads 4 > logs/budget_lorenz.log 2>&1 ;;
  *) echo "usage: bash run_cpu_jobs.sh budget_bimodal|budget_lorenz" ;;
esac
