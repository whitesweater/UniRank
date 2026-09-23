#!/usr/bin/env python3
"""Run the three HyFormer-MerRec dimension tasks against frozen r=16 evidence."""

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

from scripts.collect_sisa_component_results import parsed_value
from unirank.utils import load_config

STUDY = "sisa_score_dim"
DIRECTORY = ROOT / "experiments/ablations" / STUDY
EXPERIMENT = "HyFormer_MerRec_Action"
DIMENSIONS = (8, 32, 64)
SEED = 20262028
WORLD_SIZE = 2
BATCH_SIZE = 16384


@dataclass(frozen=True)
class DimensionTask:
    task_id: int
    score_dim: int
    attempt: int = 1
    experiment: str = EXPERIMENT

    @property
    def run_id(self) -> str:
        return (f"SISA_score_dim_{self.experiment}_r{self.score_dim}"
                f"_seed{SEED}_attempt{self.attempt}")


def study_task(task_id: int, attempt: int = 1) -> DimensionTask:
    if task_id not in range(len(DIMENSIONS)) or attempt < 1:
        raise ValueError("task_id must be 0..2 and attempt must be positive")
    return DimensionTask(task_id, DIMENSIONS[task_id], attempt)


def reference_params() -> dict:
    reference = json.loads((DIRECTORY / "configs/reference_params.json").read_text())
    return {key: parsed_value(value) for key, value in reference.items()}


def uses_activation_checkpointing(task: DimensionTask) -> bool:
    return task.task_id == 2 and task.attempt == 2


def task_world_size(task: DimensionTask) -> int:
    return 4 if task.task_id == 2 and task.attempt >= 3 else WORLD_SIZE


def task_batch_size(task: DimensionTask) -> int:
    return 32768 // task_world_size(task)


def config_directory(task: DimensionTask) -> str:
    if uses_activation_checkpointing(task):
        return "experiments/ablations/sisa_score_dim/configs/r64_checkpoint_retry"
    return "./config"


def effective_params(task: DimensionTask, params: dict) -> dict:
    return dict(params, enable_bf16=True, sisa_enabled=True, sisa_score_dim=task.score_dim,
                sisa_lambda_init=0.1, sisa_score_scale=1.0, sisa_variant="full",
                seed=SEED, batch_size=task_batch_size(task), dataloader_seed=2027,
                sisa_parameter_seed=20260822, world_size=task_world_size(task),
                distributed=True, rank=0, local_rank=0, gpu=0, model_id=task.run_id,
                gradient_checkpointing=uses_activation_checkpointing(task),
                sparse_optimizer_foreach=False, sparse_adagrad_chunk_size=16777216)


def protocol_issues(actual: dict, task: DimensionTask) -> list[str]:
    expected = dict(reference_params(), model_id=task.run_id,
                    sisa_score_dim=task.score_dim, sisa_variant="full",
                    batch_size=task_batch_size(task), world_size=task_world_size(task),
                    gradient_checkpointing=uses_activation_checkpointing(task))
    return [f"Frozen protocol mismatch: {key}" for key in sorted(actual.keys() | expected.keys())
            if key not in actual or key not in expected
            or parsed_value(actual[key]) != expected[key]]


def validate_protocol(task: DimensionTask, params: dict) -> None:
    if task != study_task(task.task_id, task.attempt):
        raise ValueError("Task differs from the fixed three-task study")
    frozen = json.loads((DIRECTORY / "configs/base_configs.json").read_text())[EXPERIMENT]
    if uses_activation_checkpointing(task):
        frozen["gradient_checkpointing"] = True
    if params != frozen:
        changed = sorted(key for key in params.keys() | frozen.keys()
                         if params.get(key) != frozen.get(key))
        raise ValueError(f"Frozen base configuration changed: {changed}")
    issues = protocol_issues(effective_params(task, params), task)
    if issues:
        raise ValueError("; ".join(issues))


def training_command(task: DimensionTask) -> list[str]:
    return [
        ".venv/bin/torchrun", "--standalone", f"--nproc_per_node={task_world_size(task)}", "run_expid.py",
        "--config", config_directory(task), "--expid", task.experiment,
        "--gpu", ",".join(str(index) for index in range(task_world_size(task))),
        "--run-id", task.run_id, "--seed", str(SEED), "--batch-size", str(task_batch_size(task)),
        "--dataloader-seed", "2027", "--sisa-parameter-seed", "20260822",
        "--sisa-enabled", "--sisa-score-dim", str(task.score_dim),
        "--sisa-lambda-init", "0.1", "--sisa-score-scale", "1.0", "--sisa-variant", "full",
        "--sparse-optimizer-foreach", "false", "--sparse-adagrad-chunk-size", "16777216",
    ]


def check_outputs(task: DimensionTask, root: Path = ROOT) -> None:
    paths = [root / "checkpoints/MerRec_Action" / (task.run_id + suffix)
             for suffix in (".log", ".model")]
    paths.append(root / "artifacts" / STUDY / task.run_id)
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; increment --attempt")


def task_manifest(task: DimensionTask, gpu_names: list[str], revision: str,
                  environment: dict) -> dict:
    manifest = dict(asdict(task), study=STUDY, run_id=task.run_id, variant="full", seed=SEED,
                world_size=task_world_size(task), per_gpu_batch=task_batch_size(task), global_batch=32768,
                dataloader_seed=2027, sisa_parameter_seed=20260822, gpu_names=gpu_names,
                job_id=environment["SLURM_JOB_ID"],
                array_job_id=environment["SLURM_ARRAY_JOB_ID"], git_revision=revision,
                command=training_command(task))
    if uses_activation_checkpointing(task):
        manifest["gradient_checkpointing"] = True
        manifest["execution_adjustment"] = "activation_checkpointing_after_r64_oom"
    if task_world_size(task) == 4:
        manifest["execution_adjustment"] = "four_gpu_same_global_batch_after_r64_oom"
    return manifest


def source_hashes() -> dict:
    paths = [Path("run_expid.py"), Path(__file__).relative_to(ROOT),
             Path("scripts/submit_sisa_score_dim.sbatch"),
             Path("config/model_config.yaml"), Path("config/dataset_config.yaml")]
    for directory in ("unirank", "model_zoo"):
        paths.extend(sorted(Path(directory).rglob("*.py")))
    paths.extend(sorted((DIRECTORY / "configs").rglob("*")))
    return {str(path.relative_to(ROOT) if path.is_absolute() else path):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    task = study_task(args.task_id, args.attempt)
    os.chdir(ROOT)
    params = load_config(config_directory(task), EXPERIMENT)
    validate_protocol(task, params)
    check_outputs(task)
    command = training_command(task)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_ARRAY_JOB_ID"):
        raise RuntimeError("Formal tasks require the Slurm array launcher")
    if int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)) != task.task_id:
        raise RuntimeError("Slurm array task ID differs from the requested task")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    world_size = task_world_size(task)
    if len(devices) != world_size or any(not device for device in devices):
        raise RuntimeError(f"Exactly {world_size} Slurm-visible GPUs are required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) < world_size * 8:
        raise RuntimeError(f"At least {world_size * 8} allocated CPUs are required")
    if int(os.environ.get("SLURM_MEM_PER_NODE", 0)) < 480 * 1024:
        raise RuntimeError("At least 480 GiB allocated host memory is required")
    gpu_names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
    ).strip().splitlines()
    if len(gpu_names) != world_size:
        raise RuntimeError(f"Expected {world_size} allocated GPUs, got {gpu_names}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = task_manifest(task, gpu_names, revision, os.environ)
    manifest["source_sha256"] = source_hashes()
    manifest["git_status"] = subprocess.check_output(
        ["git", "status", "--short"], text=True,
    ).splitlines()
    import torch
    manifest["environment"] = dict(python=sys.version, torch=torch.__version__,
                                   cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    output = ROOT / "artifacts" / STUDY / task.run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("SISA_SCORE_DIM_PROTOCOL " + json.dumps(manifest, sort_keys=True), flush=True)
    environment = dict(os.environ, PYTHONUNBUFFERED="1",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    environment.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
    with tempfile.TemporaryDirectory(prefix=f"ur-score-dim-{task.task_id}-") as temporary:
        environment.update(TMPDIR=temporary, TMP=temporary, TEMP=temporary,
                           TORCHINDUCTOR_CACHE_DIR=f"{temporary}/torchinductor",
                           TRITON_CACHE_DIR=f"{temporary}/triton",
                           TORCH_EXTENSIONS_DIR=f"{temporary}/torch_extensions")
        result = subprocess.run(command, env=environment, check=False)
    (output / "outcome.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
    if result.returncode == 0:
        print(f"SISA_SCORE_DIM_COMPLETE run_id={task.run_id}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
