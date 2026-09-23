#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --job-name=unirank-sisa-dim-memory-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --gres=gpu:2
#SBATCH --time=00:25:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for memory_budget in 0.5 0.25 0.75; do
  budget_tag=${memory_budget//./}
  export TORCHINDUCTOR_CACHE_DIR="/tmp/sisa-dim-probe-${SLURM_JOB_ID}-${budget_tag}"
  if .venv/bin/torchrun --standalone --nproc_per_node=2 scripts/run_sisa_memory_budget.py \
    --activation-memory-budget "$memory_budget" --verify-equivalence --probe-steps 3 -- \
    --config ./config --expid HyFormer_MerRec_Action --gpu 0,1 \
    --run-id "SISA_score_dim_memory_probe_budget${budget_tag}_${SLURM_JOB_ID}" \
    --seed 20262028 --batch-size 16384 --dataloader-seed 2027 --sisa-parameter-seed 20260822 \
    --sisa-enabled --sisa-score-dim 64 --sisa-lambda-init 0.1 --sisa-score-scale 1.0 \
    --sisa-variant full --sparse-optimizer-foreach false --sparse-adagrad-chunk-size 16777216; then
    echo "SISA_MEMORY_PROBE_JOB_PASS activation_memory_budget=${memory_budget}"
    exit 0
  fi
done
exit 1
