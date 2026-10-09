#!/usr/bin/env bash
#SBATCH --job-name=rise-auc-ci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=acd_u
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# Submit with sbatch from the repository root.
#SBATCH --chdir=.
#SBATCH --output=experiments/rise_auc_significance_20260926/auc-%j.out
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
task_output="experiments/rise_auc_significance_20260926/results_${SLURM_JOB_ID}"
.venv/bin/python experiments/rise_auc_significance_20260926/run_auc.py \
  --run --plan experiments/rise_auc_significance_20260926/plan.json --output "$task_output"
