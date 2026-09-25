#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --job-name=unirank-rise-prediction-smoke
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
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="/tmp/rise-prediction-${SLURM_JOB_ID}/torchinductor"
export TRITON_CACHE_DIR="/tmp/rise-prediction-${SLURM_JOB_ID}/triton"
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
    experiments/rise_sample_analysis/scripts/export_predictions.py \
    --variants B C D E --limit-batches 2 --num-workers 2 \
    --output-dir "artifacts/rise_sample_analysis/predictions/smoke_${SLURM_JOB_ID}"
