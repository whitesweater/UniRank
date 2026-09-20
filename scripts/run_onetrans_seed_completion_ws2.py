#!/usr/bin/env python3
"""Independent two-GPU alternative for the eight retained OneTrans SISA tasks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_onetrans_seed_completion as four_gpu
from unirank.utils import load_config

STUDY = "onetrans_seed_completion_ws2"
STUDY_DIR = ROOT / "experiments" / STUDY
WORLD_SIZE = 2
BATCH_SIZE = 16384


@dataclass(frozen=True)
class TwoGpuTask(four_gpu.SeedTask):
    @property
    def run_id(self):
        return (f"SISA_seed_completion_{self.experiment}_sisa_seed{self.seed}"
                f"_ws2_bs16384_acc1_attempt{self.attempt}")


def study_task(task_id: int, attempt: int = 1) -> TwoGpuTask:
    original = four_gpu.study_task(task_id, attempt)
    four_gpu.validate_task_active(original)
    return TwoGpuTask(**asdict(original))


def effective_params(task, params):
    return dict(four_gpu.effective_params(task, params), world_size=WORLD_SIZE,
                batch_size=BATCH_SIZE)


def expected_params(task):
    return dict(four_gpu.expected_params(task), world_size=WORLD_SIZE, batch_size=BATCH_SIZE)


def validate_protocol(task, params):
    four_gpu.validate_protocol(four_gpu.study_task(task.task_id, task.attempt), params)
    if effective_params(task, params) != expected_params(task):
        raise ValueError("Two-GPU protocol differs beyond documented world size and batch")


def training_command(task):
    command = four_gpu.training_command(task)
    command[command.index("--nproc_per_node=4")] = "--nproc_per_node=2"
    command[command.index("--gpu") + 1] = "0,1"
    command[command.index("--batch-size") + 1] = str(BATCH_SIZE)
    return command


def ensure_fresh_outputs(task, root=ROOT):
    paths = [root / "checkpoints" / task.dataset / f"{task.run_id}{suffix}"
             for suffix in (".log", ".model")]
    paths.append(root / "artifacts" / STUDY / task.run_id)
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; use a new attempt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    task = study_task(args.task_id, args.attempt)
    os.chdir(ROOT)
    params = load_config(str(four_gpu.CONFIG_DIR), task.experiment)
    validate_protocol(task, params)
    ensure_fresh_outputs(task)
    command = training_command(task)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Formal runs require the Slurm launcher")
    if int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")) != task.task_id:
        raise RuntimeError("Slurm task identity mismatch")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != WORLD_SIZE or not all(devices):
        raise RuntimeError("Exactly two allocated GPUs are required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 16:
        raise RuntimeError("At least 16 CPUs are required")
    if int(os.environ.get("SLURM_MEM_PER_NODE", "0")) < 480 * 1024:
        raise RuntimeError("At least 480 GiB host memory is required")
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
    ).strip().splitlines()
    if len(names) != WORLD_SIZE or any("H100" not in name for name in names):
        raise RuntimeError(f"Expected two H100 GPUs, got {names}")
    paths = [Path("run_expid.py"), Path(__file__).relative_to(ROOT),
             Path("scripts/run_onetrans_seed_completion.py"),
             Path("scripts/submit_onetrans_seed_completion_ws2.sbatch"),
             Path("config/model_config.yaml"), Path("config/dataset_config.yaml")]
    for directory, pattern in [("unirank", "*.py"), ("model_zoo", "*.py"),
                               (str(four_gpu.CONFIG_DIR.relative_to(ROOT)), "*.yaml")]:
        paths.extend(sorted(Path(directory).rglob(pattern)))
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output = ROOT / "artifacts" / STUDY / task.run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(asdict(task), study=STUDY, source_study=four_gpu.STUDY, model=task.model,
                    setting=task.setting, experiment=task.experiment, run_id=task.run_id,
                    world_size=WORLD_SIZE, per_gpu_batch=BATCH_SIZE, global_batch=32768,
                    dataloader_seed=task.dataloader_seed, sisa_parameter_seed=task.sisa_parameter_seed,
                    gpu_names=names, job_id=os.environ["SLURM_JOB_ID"],
                    array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
                    git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    source_sha256=hashes, effective_params=effective_params(task, params), command=command)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("ONETRANS_WS2_PROTOCOL " + json.dumps(manifest, sort_keys=True), flush=True)
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    environment.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
    with tempfile.TemporaryDirectory(prefix=f"ur-onetrans-ws2-{task.task_id}-") as temporary:
        environment.update(TMPDIR=temporary, TMP=temporary, TEMP=temporary,
                           TORCHINDUCTOR_CACHE_DIR=f"{temporary}/torchinductor",
                           TRITON_CACHE_DIR=f"{temporary}/triton",
                           TORCH_EXTENSIONS_DIR=f"{temporary}/torch_extensions")
        try:
            result = subprocess.run(command, env=environment, check=False)
        except Exception as error:
            (output / "outcome.json").write_text(json.dumps({"returncode": None, "error": str(error)}) + "\n")
            raise
    (output / "outcome.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
    if result.returncode == 0:
        print(f"ONETRANS_WS2_COMPLETE run_id={task.run_id}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
