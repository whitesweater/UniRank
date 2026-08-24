#!/bin/bash
#SBATCH --job-name=unirank-smoke
#SBATCH --account=lthpc
#SBATCH --partition=rtx4090
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONUNBUFFERED=1

echo "host=$(hostname) job_id=${SLURM_JOB_ID} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
.venv/bin/python scripts/unirank_gpu_smoke.py
