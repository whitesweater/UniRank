#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --job-name=unirank-sisa-single-seed-ws2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --array=0-15%8
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
cd "$repo_root"

export SISA_WORLD_SIZE=2
export SISA_BATCH_SIZE=16384
export SISA_PROTOCOL=ws2_bs16384_acc1
export SISA_SEED="${SISA_SEED:-20262028}"
export SISA_DATALOADER_SEED="${SISA_DATALOADER_SEED:-2027}"
export SISA_PARAMETER_SEED="${SISA_PARAMETER_SEED:-20260822}"
export SISA_STUDY="${SISA_STUDY:-sisa_single_seed${SISA_SEED}}"
export SISA_ATTEMPT="${SISA_ATTEMPT:-1}"

bash "$repo_root/scripts/run_sisa_single_seed_task.sh"
