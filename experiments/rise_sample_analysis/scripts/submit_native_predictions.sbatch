#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --job-name=unirank-rise-native-predictions
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
export TORCHINDUCTOR_CACHE_DIR="/tmp/rise-native-export-${SLURM_JOB_ID}/torchinductor"
export TRITON_CACHE_DIR="/tmp/rise-native-export-${SLURM_JOB_ID}/triton"
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
    experiments/rise_sample_analysis/scripts/export_predictions.py \
    --variants A --num-workers 4 --baseline-task-id 643809_0 \
    --extra-baseline-log artifacts/rise_sample_analysis/training_checkpoints/MerRec_Action/RISE_matched_HyFormer_MerRec_Action_baseline_seed20262028_attempt1.log \
    --output-dir artifacts/rise_sample_analysis/predictions/native_merrec_20260923
