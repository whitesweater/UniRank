#!/bin/bash
#SBATCH --job-name=unirank-sisa-strict
#SBATCH --account=lthpc
#SBATCH --partition=rtx4090
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:rtx4090:4
#SBATCH --time=12:00:00
#SBATCH --array=0-31%1
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONUNBUFFERED=1

models=(OneTrans HiFormer RankMixer Zenith)
datasets=(QK_Video_Action KuaiRand_Video_Action Taobao_Action MerRec_Action)

setting_index=$((SLURM_ARRAY_TASK_ID % 2))
cell_index=$((SLURM_ARRAY_TASK_ID / 2))
dataset_index=$((cell_index % ${#datasets[@]}))
model_index=$((cell_index / ${#datasets[@]}))

model="${models[$model_index]}"
dataset="${datasets[$dataset_index]}"
experiment="${model}_${dataset}"
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

# Zenith-MerRec has 2.26B embedding parameters. On PyTorch 2.8, CUDA
# Adagrad defaults to the foreach implementation and allocates an additional
# ~7.8 GiB multi-tensor temporary at the first step, which exceeds 48 GiB GPUs.
# The scalar-tensor path applies the same Adagrad update with a lower peak.
if [[ "$model" == Zenith && "$dataset" == MerRec_Action ]]; then
  arguments+=(
    --sparse-optimizer-foreach false
    --sparse-adagrad-chunk-size 16777216
  )
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

run_id="SISA_strict_${model}_${dataset}_${setting}_seed20262027"
arguments+=(--run-id "$run_id")

IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#visible_devices[@]} != 4 )); then
  echo "STRICT_PROTOCOL_ERROR expected 4 Slurm-visible GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
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
    raise SystemExit(f"STRICT_PROTOCOL_ERROR config mismatch: {errors}")
PY

temp_root="$(mktemp -d "/tmp/ur-strict-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}.XXXXXX")"
cleanup() {
  if [[ "$temp_root" == /tmp/ur-strict-* && -d "$temp_root" ]]; then
    rm -rf -- "$temp_root"
  fi
}
trap cleanup EXIT INT TERM

echo "STRICT_PROTOCOL world_size=4 per_gpu_batch=8192 global_batch=32768 accumulation_steps=1 epochs=1 seed=20262027 bf16=true"
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

echo "SISA_NATIVE_STRICT_COMPLETE run_id=${run_id}"
