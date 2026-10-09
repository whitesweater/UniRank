#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --job-name=rise-order-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
# Submit with sbatch from the repository root.
#SBATCH --chdir=.
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/artifacts/rise_history_permutation_20260925/runtime_cache/v2_${SLURM_JOB_ID}/torchinductor"
export TRITON_CACHE_DIR="$PWD/artifacts/rise_history_permutation_20260925/runtime_cache/v2_${SLURM_JOB_ID}/triton"
cases=(rankmixer_kuairand hiformer_taac25)
case_name="${cases[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required}]}"
.venv/bin/python -X faulthandler experiments/rise_history_permutation_20260925/run_two_stage_v2.py \
    --case "$case_name" \
    --output-dir "artifacts/rise_history_permutation_20260925/${case_name}/v2_job_${SLURM_JOB_ID}"
