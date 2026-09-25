#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --job-name=unirank-rise-extended-benchmark
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
export TORCHINDUCTOR_CACHE_DIR="/tmp/rise-extended-benchmark-${SLURM_JOB_ID}/torchinductor"
export TRITON_CACHE_DIR="/tmp/rise-extended-benchmark-${SLURM_JOB_ID}/triton"

rise_extended_output="artifacts/rise_sample_analysis/mechanism/extended_benchmark_${SLURM_JOB_ID}"
rise_extended_started="$(date +%s)"
rise_extended_finish() {
    rise_extended_status=$?
    .venv/bin/python - "$rise_extended_output" "$rise_extended_started" "$rise_extended_status" "$SLURM_JOB_ID" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
directory = Path(sys.argv[1])
directory.mkdir(parents=True, exist_ok=True)
result = {"job_id": sys.argv[4], "exit_status": int(sys.argv[3]),
          "job_shell_wall_seconds": time.time() - int(sys.argv[2]),
          "finished_utc": datetime.now(timezone.utc).isoformat(),
          "scheduler_accounting_remains_authoritative": True}
(directory / "job_shell_outcome.json").write_text(json.dumps(result, indent=2) + "\n")
PY
}
trap rise_extended_finish EXIT

.venv/bin/python experiments/rise_sample_analysis/scripts/run_extended_benchmark.py \
    --source-dir artifacts/rise_sample_analysis/mechanism/observed_643858 \
    --cache-dir artifacts/rise_sample_analysis/mechanism/benchmark_cache_643858 \
    --native-log artifacts/rise_sample_analysis/training_checkpoints/MerRec_Action/RISE_matched_HyFormer_MerRec_Action_baseline_seed20262028_attempt1.log \
    --output-dir "$rise_extended_output"
