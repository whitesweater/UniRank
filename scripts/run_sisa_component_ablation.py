#!/usr/bin/env python3
"""Run three RankMixer-KuaiRand B/C/D tasks matching the H100 seed-20262028 run."""

from __future__ import annotations

import argparse
import ast
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

from unirank.utils import load_config

STUDY = "sisa_components_bcd"
STUDY_DIR = ROOT / "experiments" / "ablations" / STUDY
EXPERIMENTS = ("RankMixer_KuaiRand_Video_Action",)
VARIANTS = (("B", "content_only"), ("C", "decay_only"), ("D", "rotation_only"))
SEED = 20262028
WORLD_SIZE = 2
BATCH_SIZE = 16384
DATALOADER_SEED = 2027
PARAMETER_SEED = 20260822


@dataclass(frozen=True)
class ComponentTask:
    task_id: int
    experiment: str
    setting: str
    variant: str
    attempt: int = 1

    @property
    def model(self):
        return self.experiment.split("_", 1)[0]

    @property
    def dataset(self):
        return self.experiment.split("_", 1)[1]

    @property
    def run_id(self):
        return f"SISA_components_{self.experiment}_{self.setting}_seed{SEED}_attempt{self.attempt}"


def study_task(task_id: int, attempt: int = 1) -> ComponentTask:
    if not 0 <= task_id < 3 or attempt < 1:
        raise ValueError("task_id must be 0..2 and attempt must be positive")
    setting, variant = VARIANTS[task_id % 3]
    return ComponentTask(task_id, EXPERIMENTS[task_id // 3], setting, variant, attempt)


def parse_logged_value(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def effective_params(task: ComponentTask, params: dict) -> dict:
    return dict(
        params, enable_bf16=True, sisa_enabled=True, sisa_score_dim=16,
        sisa_lambda_init=0.1, sisa_score_scale=1.0, sisa_variant=task.variant,
        seed=SEED, batch_size=BATCH_SIZE, dataloader_seed=DATALOADER_SEED,
        sisa_parameter_seed=PARAMETER_SEED, world_size=WORLD_SIZE,
        distributed=True, rank=0, local_rank=0, gpu=0, model_id=task.run_id,
    )


def validate_protocol(task: ComponentTask, params: dict) -> None:
    frozen = json.loads((STUDY_DIR / "configs" / "base_configs.json").read_text())[task.experiment]
    if params != frozen:
        changed = sorted(key for key in params.keys() | frozen.keys()
                         if params.get(key) != frozen.get(key))
        raise ValueError(f"Frozen study config changed: {changed}")
    # Compare every logged field, including loader, side information and metrics.
    reference = json.loads((STUDY_DIR / "configs" / "reference_params.json").read_text())
    reference = {key: parse_logged_value(value) for key, value in reference.items()}
    effective = effective_params(task, params)
    for key in reference.keys() | effective.keys():
        if key in {"model_id", "sisa_variant"}:
            continue
        if effective.get(key) != reference.get(key):
            raise ValueError(f"Historical reference mismatch for {key}")


def training_command(task: ComponentTask) -> list[str]:
    command = [
        ".venv/bin/torchrun", "--standalone", f"--nproc_per_node={WORLD_SIZE}", "run_expid.py",
        "--config", "./config", "--expid", task.experiment, "--gpu", "0,1",
        "--run-id", task.run_id, "--seed", str(SEED), "--batch-size", str(BATCH_SIZE),
        "--dataloader-seed", str(DATALOADER_SEED), "--sisa-parameter-seed", str(PARAMETER_SEED),
        "--sisa-enabled", "--sisa-score-dim", "16", "--sisa-lambda-init", "0.1",
        "--sisa-score-scale", "1.0", "--sisa-variant", task.variant,
    ]
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    task = study_task(args.task_id, args.attempt)
    os.chdir(ROOT)
    params = load_config("config", task.experiment)
    validate_protocol(task, params)
    command = training_command(task)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Formal tasks must be submitted through the Slurm launcher")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != WORLD_SIZE or any(not device for device in devices):
        raise RuntimeError("Exactly two Slurm-visible GPUs are required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) < 16:
        raise RuntimeError("At least 16 allocated CPUs are required")
    if int(os.environ.get("SLURM_MEM_PER_NODE", 0)) < 480 * 1024:
        raise RuntimeError("At least 480 GiB allocated host memory is required")
    for suffix in (".log", ".model"):
        path = ROOT / "checkpoints" / task.dataset / (task.run_id + suffix)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; increment --attempt")
    gpu_names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
    ).strip().splitlines()
    if len(gpu_names) != WORLD_SIZE or any("H100" not in name for name in gpu_names):
        raise RuntimeError(f"Expected 2 allocated H100 GPUs, got {gpu_names}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_paths = [Path("run_expid.py"), Path("config/model_config.yaml"),
                    Path("config/dataset_config.yaml"), Path(__file__).relative_to(ROOT)]
    for directory in ("unirank", "model_zoo"):
        source_paths.extend(sorted(Path(directory).rglob("*.py")))
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    output = ROOT / "artifacts" / STUDY / task.run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(asdict(task), run_id=task.run_id, seed=SEED, world_size=WORLD_SIZE,
                    per_gpu_batch=BATCH_SIZE, global_batch=32768, dataloader_seed=DATALOADER_SEED,
                    sisa_parameter_seed=PARAMETER_SEED, gpu_names=gpu_names,
                    job_id=os.environ["SLURM_JOB_ID"],
                    array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
                    git_revision=revision, source_sha256=hashes, command=command)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("SISA_COMPONENT_PROTOCOL " + json.dumps(manifest, sort_keys=True), flush=True)
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    environment.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
    with tempfile.TemporaryDirectory(prefix=f"ur-components-{task.task_id}-") as temporary:
        environment.update(TMPDIR=temporary, TMP=temporary, TEMP=temporary,
                           TORCHINDUCTOR_CACHE_DIR=f"{temporary}/torchinductor",
                           TRITON_CACHE_DIR=f"{temporary}/triton",
                           TORCH_EXTENSIONS_DIR=f"{temporary}/torch_extensions")
        result = subprocess.run(command, env=environment, check=False)
    (output / "outcome.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
    if result.returncode == 0:
        print(f"SISA_COMPONENT_COMPLETE run_id={task.run_id}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
