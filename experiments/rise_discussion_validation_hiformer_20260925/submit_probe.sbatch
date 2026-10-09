#!/usr/bin/env bash
#SBATCH --job-name=rise-discussion-hiformer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
# Submit with sbatch from the repository root.
#SBATCH --chdir=.
#SBATCH --output=experiments/rise_discussion_validation_hiformer_20260925/job-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="${PWD}/experiments/rise_discussion_validation_hiformer_20260925/cache-${SLURM_JOB_ID}/torchinductor"
export TRITON_CACHE_DIR="${PWD}/experiments/rise_discussion_validation_hiformer_20260925/cache-${SLURM_JOB_ID}/triton"
.venv/bin/python -X faulthandler experiments/rise_discussion_validation_hiformer_20260925/run_probe.py \
  --output-dir "experiments/rise_discussion_validation_hiformer_20260925/results_${SLURM_JOB_ID}"
