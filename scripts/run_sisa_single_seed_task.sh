#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
: "${SISA_WORLD_SIZE:?SISA_WORLD_SIZE is required}"
: "${SISA_BATCH_SIZE:?SISA_BATCH_SIZE is required}"
: "${SISA_PROTOCOL:?SISA_PROTOCOL is required}"

seed="${SISA_SEED:-20262028}"
dataloader_seed="${SISA_DATALOADER_SEED:-2027}"
sisa_parameter_seed="${SISA_PARAMETER_SEED:-20260822}"
study="${SISA_STUDY:-sisa_single_seed${seed}}"
attempt="${SISA_ATTEMPT:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

if [[ ! "$study" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "SISA_SINGLE_SEED_ERROR unsafe study identifier: $study" >&2
  exit 2
fi
expected_study="sisa_single_seed${seed}"
if [[ "$study" != "$expected_study" ]]; then
  echo "SISA_SINGLE_SEED_ERROR study/seed mismatch: study=$study expected=$expected_study" >&2
  exit 2
fi

case "$SISA_WORLD_SIZE" in
  2)
    gpu_ids="0,1"
    ;;
  4)
    gpu_ids="0,1,2,3"
    ;;
  *)
    echo "SISA_SINGLE_SEED_ERROR unsupported world size: $SISA_WORLD_SIZE" >&2
    exit 2
    ;;
esac

global_batch=$((SISA_WORLD_SIZE * SISA_BATCH_SIZE))
if (( global_batch != 32768 )); then
  echo "SISA_SINGLE_SEED_ERROR global batch mismatch: $global_batch" >&2
  exit 2
fi
minimum_cpus=$((SISA_WORLD_SIZE * 8))
minimum_memory_mb=$((SISA_WORLD_SIZE * 245760))
if (( ${SLURM_CPUS_PER_TASK:-0} < minimum_cpus )); then
  echo "SISA_SINGLE_SEED_ERROR insufficient CPUs: ${SLURM_CPUS_PER_TASK:-unset}" >&2
  exit 2
fi
if (( ${SLURM_MEM_PER_NODE:-0} < minimum_memory_mb )); then
  echo "SISA_SINGLE_SEED_ERROR insufficient memory: ${SLURM_MEM_PER_NODE:-unset} MB" >&2
  exit 2
fi

for required_dir in \
  "$repo_root/logs" \
  "$repo_root/checkpoints" \
  "$repo_root/artifacts/$study/telemetry"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "SISA_SINGLE_SEED_ERROR missing pre-created directory: $required_dir" >&2
    exit 2
  fi
done

IFS=$'\t' read -r model dataset < <(
  .venv/bin/python scripts/sisa_single_seed_tasks.py \
    --task-id "$SLURM_ARRAY_TASK_ID"
)
experiment="${model}_${dataset}"
run_id="SISA_single_seed_${model}_${dataset}_sisa_seed${seed}_${SISA_PROTOCOL}_attempt${attempt}"

checkpoint_root="$repo_root/checkpoints/$dataset"
for protected_path in \
  "$checkpoint_root/$run_id.log" \
  "$checkpoint_root/$run_id.model"; do
  if [[ -e "$protected_path" ]]; then
    echo "SISA_SINGLE_SEED_ERROR refusing to overwrite existing path: $protected_path" >&2
    exit 2
  fi
done

mapfile -t gpu_names < <(
  nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/[[:space:]]*$//'
)
if (( ${#gpu_names[@]} != SISA_WORLD_SIZE )); then
  echo "SISA_SINGLE_SEED_ERROR expected $SISA_WORLD_SIZE visible GPUs, got ${#gpu_names[@]}" >&2
  exit 2
fi
first_gpu="${gpu_names[0]}"
for gpu_name in "${gpu_names[@]}"; do
  if [[ "$gpu_name" != "$first_gpu" ]]; then
    echo "SISA_SINGLE_SEED_ERROR mixed GPU models: ${gpu_names[*]}" >&2
    exit 2
  fi
  if [[ "$gpu_name" != *H100* ]]; then
    echo "SISA_SINGLE_SEED_ERROR expected H100, got $gpu_name" >&2
    exit 2
  fi
done

.venv/bin/python - "$experiment" <<'PY'
import sys

from unirank.utils import load_config

experiment = sys.argv[1]
params = load_config("./config", experiment)
expected = {
    "accumulation_steps": 1,
    "epochs": 1,
    "max_len": 100,
}
errors = {
    key: (params.get(key), value)
    for key, value in expected.items()
    if params.get(key) != value
}
if errors:
    raise SystemExit(f"SISA_SINGLE_SEED_ERROR config mismatch: {errors}")
PY

arguments=(
  --config ./config
  --expid "$experiment"
  --gpu "$gpu_ids"
  --run-id "$run_id"
  --seed "$seed"
  --batch-size "$SISA_BATCH_SIZE"
  --dataloader-seed "$dataloader_seed"
  --sisa-parameter-seed "$sisa_parameter_seed"
  --sisa-enabled
  --sisa-score-dim 16
  --sisa-lambda-init 0.1
  --sisa-score-scale 1.0
)

if [[ "$dataset" == MerRec_Action ]]; then
  arguments+=(
    --sparse-optimizer-foreach false
    --sparse-adagrad-chunk-size 16777216
  )
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

temp_root="$(mktemp -d "/tmp/ur-single-seed-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}.XXXXXX")"
telemetry_path="$repo_root/artifacts/$study/telemetry/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SISA_PROTOCOL}_attempt${attempt}.csv"
monitor_pid=""

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  if [[ "$temp_root" == /tmp/ur-single-seed-* && -d "$temp_root" ]]; then
    rm -rf -- "$temp_root"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv \
  --loop=30 > "$telemetry_path" &
monitor_pid=$!

echo "SISA_SINGLE_SEED_HARDWARE gpu_count=${#gpu_names[@]} gpu_name=${first_gpu} host=$(hostname)"
echo "SISA_SINGLE_SEED_ALLOCATION cpus=${SLURM_CPUS_PER_TASK:-unset} memory_per_node=${SLURM_MEM_PER_NODE:-unset}"
echo "SISA_SINGLE_SEED_PROTOCOL protocol=${SISA_PROTOCOL} world_size=${SISA_WORLD_SIZE} per_gpu_batch=${SISA_BATCH_SIZE} global_batch=${global_batch} accumulation_steps=1 epochs=1 seed=${seed} dataloader_seed=${dataloader_seed} sisa_parameter_seed=${sisa_parameter_seed} bf16=true"
echo "task_id=${SLURM_ARRAY_TASK_ID} job_id=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} experiment=${experiment} setting=sisa study=${study} run_id=${run_id} attempt=${attempt} telemetry=${telemetry_path}"

TMPDIR="$temp_root" \
TMP="$temp_root" \
TEMP="$temp_root" \
TORCHINDUCTOR_CACHE_DIR="$temp_root/torchinductor" \
TRITON_CACHE_DIR="$temp_root/triton" \
TORCH_EXTENSIONS_DIR="$temp_root/torch_extensions" \
  .venv/bin/torchrun \
    --standalone \
    --nproc_per_node="$SISA_WORLD_SIZE" \
    run_expid.py \
    "${arguments[@]}"

echo "SISA_SINGLE_SEED_COMPLETE task_id=${SLURM_ARRAY_TASK_ID} run_id=${run_id} protocol=${SISA_PROTOCOL}"
