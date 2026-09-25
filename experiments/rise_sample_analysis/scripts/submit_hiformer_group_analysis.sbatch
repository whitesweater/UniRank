#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --job-name=unirank-rise-hiformer-groups
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
rise_dataset="${1:?dataset required}"
rise_frequency_dir="${2:?frequency directory required}"
case "$rise_dataset" in
    QK_Video_Action|TencentGR_10M_Action) ;;
    *) exit 2 ;;
esac
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
.venv/bin/python experiments/rise_sample_analysis/scripts/analyze_hiformer_groups.py analyze \
    --dataset "$rise_dataset" \
    --baseline-dir "artifacts/rise_sample_analysis/predictions/hiformer_${rise_dataset}_20260923/A" \
    --rise-dir "artifacts/rise_sample_analysis/predictions/hiformer_${rise_dataset}_20260923/E" \
    --frequency-dir "$rise_frequency_dir" \
    --output-dir "artifacts/rise_sample_analysis/hiformer_group_analysis/${rise_dataset}/job_${SLURM_JOB_ID}"
