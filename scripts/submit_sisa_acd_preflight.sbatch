#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --job-name=unirank-sisa-acd-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
cd "$repo_root"

echo "runtime_identity=$(id)"
echo "repo_root=$repo_root"
stat -c 'runtime_path owner=%U group=%G mode=%a path=%n' \
  "$repo_root" "$repo_root/logs" "$repo_root/artifacts/sisa_expansion_acd"
write_probe="$(mktemp "$repo_root/artifacts/sisa_expansion_acd/write-probe.XXXXXX")"
rm -f -- "$write_probe"
echo "ACD_PREFLIGHT_SHARED_WRITE=ok"

echo "host=$(hostname) job_id=${SLURM_JOB_ID:-unset} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

.venv/bin/python - <<'PY'
from pathlib import Path

import pyarrow.parquet as pq
import torch

if not torch.cuda.is_available():
    raise SystemExit("ACD_PREFLIGHT_ERROR CUDA is unavailable")
if torch.cuda.device_count() != 4:
    raise SystemExit(
        f"ACD_PREFLIGHT_ERROR expected 4 visible GPUs, got {torch.cuda.device_count()}"
    )

gpu_names = [torch.cuda.get_device_name(index) for index in range(4)]
if len(set(gpu_names)) != 1:
    raise SystemExit(f"ACD_PREFLIGHT_ERROR mixed GPU models: {gpu_names}")

datasets = (
    "QK_Video_Action",
    "KuaiRand_Video_Action",
    "TencentGR_10M_Action",
    "Taobao_Action",
    "MerRec_Action",
)
for dataset in datasets:
    files = sorted(Path("datasets", dataset, "train", "data").glob("*.parquet"))
    if not files:
        raise SystemExit(f"ACD_PREFLIGHT_ERROR missing train parquet for {dataset}")
    pq.ParquetFile(files[0]).metadata

print(f"ACD_PREFLIGHT_GPU_NAME={gpu_names[0]}")
print(f"ACD_PREFLIGHT_TORCH={torch.__version__}")
print(f"ACD_PREFLIGHT_CUDA_RUNTIME={torch.version.cuda}")
print("SISA_ACD_PREFLIGHT_COMPLETE")
PY

.venv/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/acd_nccl_preflight.py
