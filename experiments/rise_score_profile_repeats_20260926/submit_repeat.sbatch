#!/usr/bin/env bash
#SBATCH --job-name=rise-score-repeat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# Submit with sbatch from the repository root.
#SBATCH --chdir=.
#SBATCH --output=experiments/rise_score_profile_repeats_20260926/job-%j.out

set -euo pipefail
PLAN_PATH="${1:?Supply the absolute frozen plan JSON path}"
STUDY_ROOT="${PWD}/experiments/rise_score_profile_repeats_20260926"
REPEAT_PYTHON="${PWD}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="${STUDY_ROOT}/cache-${SLURM_JOB_ID}/torchinductor"
export TRITON_CACHE_DIR="${STUDY_ROOT}/cache-${SLURM_JOB_ID}/triton"
"${REPEAT_PYTHON}" -X faulthandler "${STUDY_ROOT}/run_repeat.py" --plan "${PLAN_PATH}"
"${REPEAT_PYTHON}" - "${PLAN_PATH}" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
plan = json.loads(Path(sys.argv[1]).read_text())
subprocess.run([sys.executable, str(Path(plan['old_source_dir']) / 'analyze_probe.py'),
                plan['output_dir']], check=True)
PY
