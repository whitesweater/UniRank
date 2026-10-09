#!/usr/bin/env bash
#SBATCH --job-name=rise-history-stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
# Submit with sbatch from the repository root.
#SBATCH --chdir=.
#SBATCH --output=experiments/rise_history_permutation_analysis_20260925/stats-%j.out
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo "Arguments: case new_results historical_results frozen_plan_sha256 new_output" >&2
  exit 2
fi
task_case="$1"
task_results="$2"
task_historical="$3"
task_plan_sha="$4"
task_output="$5"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
.venv/bin/python experiments/rise_history_permutation_analysis_20260925/analyze_permutation.py \
  "$task_results" --case "$task_case" --historical-results "$task_historical" \
  --expected-plan-sha256 "$task_plan_sha" --output "$task_output"
