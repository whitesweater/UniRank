#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --job-name=unirank-rise-hiformer-predictions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
rise_dataset="${1:?dataset required}"
rise_baseline_task="${2:?baseline task required}"
case "$rise_dataset" in
    QK_Video_Action|TencentGR_10M_Action) ;;
    *) exit 2 ;;
esac
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="/tmp/rise-hiformer-export-${SLURM_JOB_ID}/torchinductor"
export TRITON_CACHE_DIR="/tmp/rise-hiformer-export-${SLURM_JOB_ID}/triton"
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
    experiments/rise_sample_analysis/scripts/export_hiformer_predictions.py \
    --dataset "$rise_dataset" --variants A E --num-workers 4 \
    --extra-baseline-log "artifacts/rise_sample_analysis/training_checkpoints/${rise_dataset}/RISE_matched_HiFormer_${rise_dataset}_baseline_seed20262028_attempt1.log" \
    --baseline-task-id "$rise_baseline_task" \
    --output-dir "artifacts/rise_sample_analysis/predictions/hiformer_${rise_dataset}_20260923"
