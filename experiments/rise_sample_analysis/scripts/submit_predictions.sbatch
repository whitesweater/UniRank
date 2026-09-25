#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --job-name=unirank-rise-paired-predictions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:2
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="/tmp/rise-prediction-${SLURM_JOB_ID}/torchinductor"
export TRITON_CACHE_DIR="/tmp/rise-prediction-${SLURM_JOB_ID}/triton"
timeout 30 nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
    experiments/rise_sample_analysis/scripts/export_predictions.py \
    --variants B C D E --num-workers 4 \
    --output-dir artifacts/rise_sample_analysis/predictions/formal_20260923
