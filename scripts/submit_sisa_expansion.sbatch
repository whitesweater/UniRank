#!/bin/bash
#SBATCH --job-name=unirank-sisa-expansion
#SBATCH --account=lthpc
#SBATCH --partition=rtx4090
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:rtx4090:4
#SBATCH --time=12:00:00
#SBATCH --array=0-37%1
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONUNBUFFERED=1

# Pending cells after the completed 4x4 native-attention study:
#   - the original four models on TencentGR;
#   - UniMixer, HyFormer, and UltraHSTU on all five datasets.
# Each logical cell has adjacent baseline and SISA tasks.
experiments=(
  OneTrans_TencentGR_10M_Action
  HiFormer_TencentGR_10M_Action
  RankMixer_TencentGR_10M_Action
  Zenith_TencentGR_10M_Action
  UniMixer_QK_Video_Action
  UniMixer_KuaiRand_Video_Action
  UniMixer_TencentGR_10M_Action
  UniMixer_Taobao_Action
  UniMixer_MerRec_Action
  HyFormer_QK_Video_Action
  HyFormer_KuaiRand_Video_Action
  HyFormer_TencentGR_10M_Action
  HyFormer_Taobao_Action
  HyFormer_MerRec_Action
  UltraHSTU_QK_Video_Action
  UltraHSTU_KuaiRand_Video_Action
  UltraHSTU_TencentGR_10M_Action
  UltraHSTU_Taobao_Action
  UltraHSTU_MerRec_Action
)

setting_index=$((SLURM_ARRAY_TASK_ID % 2))
experiment_index=$((SLURM_ARRAY_TASK_ID / 2))
experiment="${experiments[$experiment_index]}"
model="${experiment%%_*}"
dataset="${experiment#*_}"
setting=baseline
arguments=(
  --config ./config
  --expid "$experiment"
  --gpu 0,1,2,3
)

if (( setting_index == 1 )); then
  setting=sisa
  arguments+=(
    --sisa-enabled
    --sisa-score-dim 16
    --sisa-lambda-init 0.1
    --sisa-score-scale 1.0
  )
fi

# MerRec has the largest embedding tables. The scalar/chunked Adagrad path is
# update-equivalent and avoids the foreach optimizer's large first-step peak.
if [[ "$dataset" == MerRec_Action ]]; then
  arguments+=(
    --sparse-optimizer-foreach false
    --sparse-adagrad-chunk-size 16777216
  )
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

run_id="SISA_expansion_${model}_${dataset}_${setting}_seed20262027"
arguments+=(--run-id "$run_id")

IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#visible_devices[@]} != 4 )); then
  echo "EXPANSION_PROTOCOL_ERROR expected 4 Slurm-visible GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 2
fi

.venv/bin/python - "$experiment" <<'PY'
import sys

from unirank.utils import load_config

experiment = sys.argv[1]
params = load_config("./config", experiment)
expected = {
    "batch_size": 8192,
    "accumulation_steps": 1,
    "epochs": 1,
    "seed": 20262027,
    "max_len": 100,
}
errors = {
    key: (params.get(key), value)
    for key, value in expected.items()
    if params.get(key) != value
}
if errors:
    raise SystemExit(f"EXPANSION_PROTOCOL_ERROR config mismatch: {errors}")
PY

temp_root="$(mktemp -d "/tmp/ur-expansion-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}.XXXXXX")"
cleanup() {
  if [[ "$temp_root" == /tmp/ur-expansion-* && -d "$temp_root" ]]; then
    rm -rf -- "$temp_root"
  fi
}
trap cleanup EXIT INT TERM

echo "EXPANSION_PROTOCOL world_size=4 per_gpu_batch=8192 global_batch=32768 accumulation_steps=1 epochs=1 seed=20262027 bf16=true"
echo "host=$(hostname) job_id=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} cuda_visible_devices=${CUDA_VISIBLE_DEVICES} experiment=${experiment} setting=${setting} run_id=${run_id}"

TMPDIR="$temp_root" \
TMP="$temp_root" \
TEMP="$temp_root" \
TORCHINDUCTOR_CACHE_DIR="$temp_root/torchinductor" \
TRITON_CACHE_DIR="$temp_root/triton" \
TORCH_EXTENSIONS_DIR="$temp_root/torch_extensions" \
  .venv/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    run_expid.py \
    "${arguments[@]}"

echo "SISA_EXPANSION_COMPLETE run_id=${run_id}"
