#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --job-name=unirank-sisa-acd-gate
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

.venv/bin/python -m scripts.audit_sisa_expansion_baselines \
  --task-id 0 \
  --task-id 8 \
  --task-id 18 \
  --task-id 28 \
  --auc-tolerance 0.01 \
  --output artifacts/sisa_expansion_acd/calibration_baseline_audit.csv

echo "SISA_ACD_CALIBRATION_GATE_COMPLETE"
