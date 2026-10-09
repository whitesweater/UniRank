#!/usr/bin/env bash
#SBATCH --job-name=rise-hiformer-stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
# Submit with sbatch from the repository root.
#SBATCH --chdir=.
#SBATCH --output=experiments/rise_discussion_validation_hiformer_20260925/stats-%j.out
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
.venv/bin/python experiments/rise_discussion_validation_hiformer_20260925/analyze_probe.py \
  experiments/rise_discussion_validation_hiformer_20260925/results_651691
