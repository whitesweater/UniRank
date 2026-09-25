#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --job-name=unirank-rise-state-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="/tmp/rise-states-${SLURM_JOB_ID}/torchinductor"
export TRITON_CACHE_DIR="/tmp/rise-states-${SLURM_JOB_ID}/triton"
.venv/bin/python -X faulthandler experiments/rise_sample_analysis/scripts/capture_compiled_control_states.py \
    --reference-dir artifacts/rise_sample_analysis/mechanism/observed_643858 \
    --output-dir "artifacts/rise_sample_analysis/mechanism/states_E_probe_${SLURM_JOB_ID}" \
    --variant E --limit-batches 1
