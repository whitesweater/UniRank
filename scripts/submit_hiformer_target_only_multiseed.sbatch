#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --job-name=rise-hiformer-target-multiseed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --array=4-7
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%A_%a.out

# Seed27: sbatch --array=0-3 --gres=gpu:4 --cpus-per-task=32 --mem=960G this-file
# Seed29: sbatch --array=4-7 this-file
# Neither array has an imposed concurrency limit; the runner checks each budget.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if [[ ! "$task_id" =~ ^[0-7]$ ]]; then
    echo "Invalid HiFormer multiseed task: $task_id" >&2
    exit 2
fi
exec .venv/bin/python scripts/run_hiformer_target_only_multiseed.py \
    --task-id "$task_id" --attempt "${RISE_HIFORMER_TARGET_ONLY_MULTISEED_ATTEMPT:-1}"
