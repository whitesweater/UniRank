#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --job-name=unirank-rise-matched-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --array=0-2%1
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
export PYTHONUNBUFFERED=1
.venv/bin/python experiments/rise_sample_analysis/scripts/run_matched_baseline.py \
    --task-id "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" \
    --attempt "${RISE_BASELINE_ATTEMPT:-1}"
