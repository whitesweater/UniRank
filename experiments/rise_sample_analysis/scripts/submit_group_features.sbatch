#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --job-name=unirank-rise-group-features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
.venv/bin/python experiments/rise_sample_analysis/scripts/build_group_features.py count-train
.venv/bin/python experiments/rise_sample_analysis/scripts/build_group_features.py \
    --output-dir "artifacts/rise_sample_analysis/group_features_smoke_${SLURM_JOB_ID}" \
    build --frequency-dir artifacts/rise_sample_analysis/group_features/frequencies \
    --part-id 0 --limit-rows 32
.venv/bin/python experiments/rise_sample_analysis/scripts/build_group_features.py build
