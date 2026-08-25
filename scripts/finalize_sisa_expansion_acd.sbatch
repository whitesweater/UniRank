#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --job-name=unirank-sisa-acd-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
cd "$repo_root"

: "${CALIBRATION_JOB_ID:?CALIBRATION_JOB_ID is required}"
: "${MAIN_JOB_ID:?MAIN_JOB_ID is required}"

.venv/bin/python -m scripts.audit_sisa_expansion_baselines \
  --auc-tolerance 0.01 \
  --output experiments/sisa_expansion_acd/results/baseline_audit.csv

override_args=()
for override in ${TASK_JOB_OVERRIDES:-}; do
  override_args+=(--job-override "$override")
done

.venv/bin/python -m scripts.collect_sisa_expansion_acd_results \
  --calibration-job "$CALIBRATION_JOB_ID" \
  --main-job "$MAIN_JOB_ID" \
  "${override_args[@]}" \
  --output-dir experiments/sisa_expansion_acd/results

echo "SISA_ACD_EXPANSION_FINAL_COMPLETE calibration_job=${CALIBRATION_JOB_ID} main_job=${MAIN_JOB_ID}"
