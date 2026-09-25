#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --job-name=unirank-rise-native-group-effects
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
.venv/bin/python experiments/rise_sample_analysis/scripts/analyze_groups.py \
    --prediction-dir artifacts/rise_sample_analysis/predictions/formal_20260923 \
    --baseline-dir artifacts/rise_sample_analysis/predictions/native_merrec_20260923/A \
    --output-dir experiments/rise_sample_analysis/results/group_analysis_native_v1
