#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --job-name=rise-hiformer-target-only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if [[ "$task_id" != 0 && "$task_id" != 1 && "$task_id" != 2 ]]; then
    echo "Invalid HiFormer extension task: $task_id" >&2
    exit 2
fi
exec .venv/bin/python scripts/run_hiformer_target_only_extension.py \
    --task-id "$task_id" --attempt "${RISE_HIFORMER_TARGET_ONLY_ATTEMPT:-1}"
