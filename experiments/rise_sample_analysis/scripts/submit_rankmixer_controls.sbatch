#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --job-name=rise-rankmixer-controls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --array=0-1%1
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
controls=(local fixed)
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if [[ "$task_id" != 0 && "$task_id" != 1 ]]; then
    echo "Invalid control task: $task_id" >&2
    exit 2
fi
.venv/bin/python experiments/rise_sample_analysis/scripts/run_rankmixer_controls.py \
    --control "${controls[$task_id]}" --attempt "${RISE_CONTROL_ATTEMPT:-1}"
