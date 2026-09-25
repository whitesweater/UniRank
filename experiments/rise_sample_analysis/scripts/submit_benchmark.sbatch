#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --job-name=unirank-rise-benchmark
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
export TORCHINDUCTOR_CACHE_DIR="/tmp/rise-benchmark-${SLURM_JOB_ID}/torchinductor"
export TRITON_CACHE_DIR="/tmp/rise-benchmark-${SLURM_JOB_ID}/triton"
.venv/bin/python experiments/rise_sample_analysis/scripts/benchmark_inference.py \
    --sample-cache artifacts/rise_sample_analysis/mechanism/observed_643858/sample_batches.pt \
    --mode compiled --variant-order B E --round-id forward01 --blocks 10 --repeats 100 \
    --output-dir "artifacts/rise_sample_analysis/mechanism/bench64_forward_${SLURM_JOB_ID}"
.venv/bin/python experiments/rise_sample_analysis/scripts/benchmark_inference.py \
    --sample-cache artifacts/rise_sample_analysis/mechanism/observed_643858/sample_batches.pt \
    --mode compiled --variant-order E B --round-id reverse01 --blocks 10 --repeats 100 \
    --output-dir "artifacts/rise_sample_analysis/mechanism/bench64_reverse_${SLURM_JOB_ID}"
