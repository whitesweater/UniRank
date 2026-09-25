#!/usr/bin/env python3
"""Run precisely one frozen RankMixer--KuaiRand local/fixed control on 2 H100s."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from unirank.utils import load_config

from controls import patched_build

STUDY = "rise_rankmixer_controls"
STUDY_DIR = ROOT / "experiments/rise_sample_analysis"
EXPERIMENT = "RankMixer_KuaiRand_Video_Action"
DATASET = "KuaiRand_Video_Action"
SEED, WORLD_SIZE, BATCH_SIZE = 20262028, 2, 16384


def run_id(control, attempt):
    if control not in {"local", "fixed"} or attempt < 1:
        raise ValueError("Invalid control or attempt")
    return f"RISE_controls_{EXPERIMENT}_{control}_seed{SEED}_attempt{attempt}"


def effective_params(params, control, attempt):
    return dict(params, enable_bf16=True, sisa_enabled=True, sisa_score_dim=16,
                sisa_lambda_init=0.1, sisa_score_scale=1.0, seed=SEED,
                batch_size=BATCH_SIZE, dataloader_seed=2027,
                sisa_parameter_seed=20260822, world_size=WORLD_SIZE,
                distributed=True, rank=0, local_rank=0, gpu=0,
                model_id=run_id(control, attempt))


def validate_protocol(control, attempt):
    params = load_config(str(ROOT / "config"), EXPERIMENT)
    frozen = json.loads((STUDY_DIR / "configs/rankmixer_controls_base_configs.json")
                        .read_text())[EXPERIMENT]
    if params != frozen:
        changed = sorted(k for k in params.keys() | frozen.keys()
                         if params.get(k) != frozen.get(k))
        raise ValueError(f"Frozen base config changed: {changed}")
    reference = json.loads((STUDY_DIR / "configs/rankmixer_controls_reference_params.json")
                           .read_text())
    for key, value in reference.items():
        try:
            reference[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    effective = effective_params(params, control, attempt)
    differences = [k for k in reference.keys() | effective.keys()
                   if k != "model_id" and reference.get(k) != effective.get(k)]
    if differences:
        raise ValueError(f"Full E historical protocol mismatch: {sorted(differences)}")
    return effective


def run_expid_arguments(control, attempt):
    return [str(ROOT / "run_expid.py"), "--config", "./config", "--expid", EXPERIMENT,
            "--gpu", "0,1", "--run-id", run_id(control, attempt), "--seed", str(SEED),
            "--batch-size", str(BATCH_SIZE), "--dataloader-seed", "2027",
            "--sisa-parameter-seed", "20260822", "--sisa-enabled",
            "--sisa-score-dim", "16", "--sisa-lambda-init", "0.1",
            "--sisa-score-scale", "1.0"]


def training_command(control, attempt):
    return [str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nproc_per_node=2",
            str(Path(__file__).resolve()), "--worker", "--control", control,
            "--attempt", str(attempt)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("local", "fixed"), required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    effective = validate_protocol(args.control, args.attempt)
    command = training_command(args.control, args.attempt)
    if args.dry_run:
        print(json.dumps({"control": args.control, "effective_params": effective,
                          "launch": shlex.join(command),
                          "underlying_run_expid": run_expid_arguments(args.control, args.attempt)},
                         indent=2))
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Formal controls require a Slurm allocation")
    if args.worker:
        if int(os.environ.get("WORLD_SIZE", 0)) != WORLD_SIZE:
            raise RuntimeError("Worker requires WORLD_SIZE=2")
        print("RISE_CONTROL_WORKER " + json.dumps({"control": args.control,
              "rank": int(os.environ.get("RANK", -1)), "score_dim": 16}), flush=True)
        sys.argv = run_expid_arguments(args.control, args.attempt)
        with patched_build(args.control):
            runpy.run_path(str(ROOT / "run_expid.py"), run_name="__main__")
        return 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != WORLD_SIZE or any(not device for device in visible):
        raise RuntimeError("Exactly 2 allocated visible GPUs are required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) < 16:
        raise RuntimeError("At least 16 allocated CPUs are required")
    if int(os.environ.get("SLURM_MEM_PER_NODE", 0)) < 480 * 1024:
        raise RuntimeError("At least 480 GiB allocated host RAM is required")
    gpu_names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()
    if len(gpu_names) != WORLD_SIZE or any("H100" not in name for name in gpu_names):
        raise RuntimeError(f"Expected two H100 GPUs, got {gpu_names}")
    identifier = run_id(args.control, args.attempt)
    for suffix in (".log", ".model"):
        path = ROOT / "checkpoints" / DATASET / (identifier + suffix)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    output = ROOT / "artifacts" / STUDY / identifier
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [ROOT / "run_expid.py", Path(__file__).resolve(),
                    Path(__file__).with_name("controls.py"),
                    ROOT / "config/model_config.yaml", ROOT / "config/dataset_config.yaml"]
    source_paths += sorted((STUDY_DIR / "configs").glob("rankmixer_controls_*.json"))
    for directory in ("unirank", "model_zoo"):
        source_paths += sorted((ROOT / directory).rglob("*.py"))
    manifest = {"study": STUDY, "control": args.control, "run_id": identifier,
                "attempt": args.attempt, "effective_params": effective,
                "reference_job_task": "548166_7", "world_size": WORLD_SIZE,
                "per_gpu_batch": BATCH_SIZE, "global_batch": WORLD_SIZE * BATCH_SIZE,
                "gpu_names": gpu_names, "job_id": os.environ["SLURM_JOB_ID"],
                "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "git_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in source_paths}, "command": command,
                "protocol_delta": "auxiliary control definition and run identifier only",
                "control_semantics": ("token-local controls, no scan, fixed zero centering gauge"
                                      if args.control == "local" else
                                      "learned input-independent constant increments with original scan"),
                "checkpoint_restore_requires": "construct with the same study control before load"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    environment.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
    with tempfile.TemporaryDirectory(prefix="rise-control-") as temporary:
        environment.update(TMPDIR=temporary, TMP=temporary, TEMP=temporary,
                           TORCHINDUCTOR_CACHE_DIR=f"{temporary}/torchinductor",
                           TRITON_CACHE_DIR=f"{temporary}/triton",
                           TORCH_EXTENSIONS_DIR=f"{temporary}/torch_extensions")
        result = subprocess.run(command, env=environment, check=False)
    (output / "outcome.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
    if result.returncode == 0:
        print(f"RISE_CONTROL_COMPLETE run_id={identifier}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
