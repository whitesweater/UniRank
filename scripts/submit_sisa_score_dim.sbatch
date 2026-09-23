#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --job-name=unirank-sisa-score-dim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --array=0-2%2
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
.venv/bin/python scripts/run_sisa_score_dim.py \
  --task-id "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" \
  --attempt "${SISA_SCORE_DIM_ATTEMPT:-1}"
