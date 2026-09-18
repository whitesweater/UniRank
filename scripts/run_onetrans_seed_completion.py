#!/usr/bin/env python3
"""Run only the ten missing OneTrans full-SISA seed repetitions."""

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

STUDY = "onetrans_seed_completion"
STUDY_DIR = ROOT / "experiments" / STUDY
CONFIG_DIR = STUDY_DIR / "configs/runtime"
DATASETS = ("QK_Video_Action", "KuaiRand_Video_Action", "TencentGR_10M_Action",
            "Taobao_Action", "MerRec_Action")
WORLD_SIZE = 4
BATCH_SIZE = 8192


@dataclass(frozen=True)
class SeedTask:
    task_id: int
    dataset: str
    seed: int
    attempt: int = 1

    @property
    def model(self):
        return "OneTrans"

    @property
    def setting(self):
        return "sisa"

    @property
    def experiment(self):
        return f"{self.model}_{self.dataset}"

    @property
    def dataloader_seed(self):
        return self.seed - 20260001

    @property
    def sisa_parameter_seed(self):
        return self.seed - 1206

    @property
    def run_id(self):
        return (f"SISA_seed_completion_{self.experiment}_sisa_seed{self.seed}"
                f"_ws4_bs8192_acc1_attempt{self.attempt}")


def study_task(task_id: int, attempt: int = 1) -> SeedTask:
    if not 0 <= task_id < 10 or attempt < 1:
        raise ValueError("task_id must be 0..9 and attempt must be positive")
    return SeedTask(task_id, DATASETS[task_id // 2], 20262028 + task_id % 2, attempt)


def parsed_value(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def expected_params(task: SeedTask) -> dict:
    references = json.loads((STUDY_DIR / "configs/reference_params.json").read_text())
    reference = references[f"{task.experiment}_sisa"]
    expected = {key: parsed_value(value) for key, value in reference.items()}
    expected.update(model_id=task.run_id, seed=task.seed,
                    dataloader_seed=task.dataloader_seed,
                    sisa_parameter_seed=task.sisa_parameter_seed)
    return expected


def effective_params(task: SeedTask, params: dict) -> dict:
    return dict(params, enable_bf16=True, sisa_enabled=True, sisa_score_dim=16,
                sisa_lambda_init=0.1, sisa_score_scale=1.0, seed=task.seed,
                dataloader_seed=task.dataloader_seed, sisa_parameter_seed=task.sisa_parameter_seed,
                world_size=WORLD_SIZE, batch_size=BATCH_SIZE, distributed=True,
                rank=0, local_rank=0, gpu=0, model_id=task.run_id)


def validate_protocol(task: SeedTask, params: dict) -> None:
    frozen = json.loads((STUDY_DIR / "configs/base_configs.json").read_text())[task.experiment]
    current = load_config(str(ROOT / "config"), task.experiment)
    if current != frozen:
        raise ValueError("Repository base config changed after this study was frozen")
    runtime_expected = dict(frozen)
    runtime_expected["group_id"] = expected_params(task)["group_id"]
    if params != runtime_expected:
        raise ValueError("Frozen runtime config changed")
    actual, expected = effective_params(task, params), expected_params(task)
    differences = [key for key in actual.keys() | expected.keys()
                   if key not in actual or key not in expected or actual[key] != expected[key]]
    if differences:
        raise ValueError(f"Historical protocol differs beyond seed bundle/run ID: {sorted(differences)}")


def training_command(task: SeedTask) -> list[str]:
    return [".venv/bin/torchrun", "--standalone", "--nproc_per_node=4", "run_expid.py",
            "--config", str(CONFIG_DIR.relative_to(ROOT)), "--expid", task.experiment,
            "--gpu", "0,1,2,3", "--run-id", task.run_id,
            "--seed", str(task.seed), "--batch-size", str(BATCH_SIZE),
            "--dataloader-seed", str(task.dataloader_seed),
            "--sisa-parameter-seed", str(task.sisa_parameter_seed),
            "--sisa-enabled", "--sisa-score-dim", "16",
            "--sisa-lambda-init", "0.1", "--sisa-score-scale", "1.0"]


def ensure_fresh_outputs(task: SeedTask, root: Path = ROOT) -> None:
    paths = [root / "checkpoints" / task.dataset / f"{task.run_id}{suffix}"
             for suffix in (".log", ".model")]
    paths.append(root / "artifacts" / STUDY / task.run_id)
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; use a new attempt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    task = study_task(args.task_id, args.attempt)
    os.chdir(ROOT)
    params = load_config(str(CONFIG_DIR), task.experiment)
    validate_protocol(task, params)
    ensure_fresh_outputs(task)
    command = training_command(task)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Formal runs require the Slurm launcher")
    if int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")) != task.task_id:
        raise RuntimeError("Slurm task identity does not match requested task")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != WORLD_SIZE or any(not device for device in devices):
        raise RuntimeError("Exactly four allocated GPUs are required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 32:
        raise RuntimeError("At least 32 CPUs are required")
    if int(os.environ.get("SLURM_MEM_PER_NODE", "0")) < 960 * 1024:
        raise RuntimeError("At least 960 GiB host memory is required")
    gpu_names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
    ).strip().splitlines()
    if len(gpu_names) != WORLD_SIZE or any("H100" not in n for n in gpu_names):
        raise RuntimeError(f"Expected four H100 GPUs, got {gpu_names}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    paths = [Path("run_expid.py"), Path(__file__).relative_to(ROOT),
             Path("scripts/submit_onetrans_seed_completion.sbatch"),
             Path("config/model_config.yaml"), Path("config/dataset_config.yaml")]
    for directory, pattern in [("unirank", "*.py"), ("model_zoo", "*.py"),
                               (str(CONFIG_DIR.relative_to(ROOT)), "*.yaml")]:
        paths.extend(sorted(Path(directory).rglob(pattern)))
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output = ROOT / "artifacts" / STUDY / task.run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(asdict(task), study=STUDY, model=task.model, setting=task.setting,
                    experiment=task.experiment, run_id=task.run_id, world_size=WORLD_SIZE,
                    per_gpu_batch=BATCH_SIZE, global_batch=32768,
                    dataloader_seed=task.dataloader_seed, sisa_parameter_seed=task.sisa_parameter_seed,
                    gpu_names=gpu_names, job_id=os.environ["SLURM_JOB_ID"],
                    array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
                    git_revision=revision, source_sha256=hashes, command=command,
                    effective_params=effective_params(task, params))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("ONETRANS_SEED_PROTOCOL " + json.dumps(manifest, sort_keys=True), flush=True)
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    environment.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
    with tempfile.TemporaryDirectory(prefix=f"ur-onetrans-{task.task_id}-") as temporary:
        environment.update(TMPDIR=temporary, TMP=temporary, TEMP=temporary,
                           TORCHINDUCTOR_CACHE_DIR=f"{temporary}/torchinductor",
                           TRITON_CACHE_DIR=f"{temporary}/triton",
                           TORCH_EXTENSIONS_DIR=f"{temporary}/torch_extensions")
        try:
            result = subprocess.run(command, env=environment, check=False)
            returncode = result.returncode
        except Exception as error:
            (output / "outcome.json").write_text(json.dumps({"returncode": None, "error": str(error)}) + "\n")
            raise
    (output / "outcome.json").write_text(json.dumps({"returncode": returncode}) + "\n")
    if returncode == 0:
        print(f"ONETRANS_SEED_COMPLETE run_id={task.run_id}", flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
