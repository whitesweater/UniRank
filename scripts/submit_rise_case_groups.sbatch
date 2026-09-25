#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --job-name=unirank-rise-case-groups
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="/tmp/rise-case-groups-${SLURM_JOB_ID}/torchinductor"
export TRITON_CACHE_DIR="/tmp/rise-case-groups-${SLURM_JOB_ID}/triton"
.venv/bin/python -X faulthandler scripts/probe_rise_case_groups.py \
    --groups-json "${1:?Supply the frozen groups JSON as the first sbatch argument}" \
    --output-dir "artifacts/rise_attention_case_search/probe_${SLURM_JOB_ID}"
