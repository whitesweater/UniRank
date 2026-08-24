#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p logs artifacts/sisa_expansion_l40

allocation_time="${SISA_ALLOCATION_TIME:-7-00:00:00}"

exec salloc \
  --job-name=unirank-sisa-l40-all \
  --account=lthpc \
  --qos=long \
  --partition=l40s \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=32 \
  --mem=256G \
  --gres=gpu:l40s:4 \
  --time="$allocation_time" \
  --signal=TERM@300 \
  bash "$repo_root/scripts/run_sisa_expansion_l40_all.sh"
