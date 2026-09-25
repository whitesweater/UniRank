"""Train only missing native baselines against frozen seed-20262028 RISE runs.

The original model/data code and all score-dimension studies remain untouched.
Reference log parameters are used directly, with three explicitly allowed changes:
disable SISA, give this run a unique identifier, and isolate its output directory.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "experiments/rise_sample_analysis"
ARTIFACTS = ROOT / "artifacts/rise_sample_analysis"
TASKS = (("HyFormer", "MerRec_Action"), ("HiFormer", "QK_Video_Action"),
         ("HiFormer", "TencentGR_10M_Action"))


def reference_log(task):
    model, dataset = task
    return ROOT / "checkpoints" / dataset / (
        f"SISA_single_seed_{model}_{dataset}_sisa_seed20262028_ws2_bs16384_acc1_attempt1.log"
    )


def reference_params(task):
    text = reference_log(task).read_text()
    raw, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
    parsed = {}
    for key, value in raw.items():
        try:
            parsed[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed[key] = value
    assert parsed["sisa_enabled"] is True
    assert (parsed["seed"], parsed["dataloader_seed"], parsed["world_size"],
            parsed["batch_size"], parsed["epochs"], parsed["max_len"]) == (
                20262028, 2027, 2, 16384, 1, 100)
    return parsed


def baseline_params(task, attempt):
    if attempt < 1:
        raise ValueError("Attempt must be positive")
    params = reference_params(task)
    model, dataset = task
    params.update(
        sisa_enabled=False,
        model_id=f"RISE_matched_{model}_{dataset}_baseline_seed20262028_attempt{attempt}",
        model_root="./artifacts/rise_sample_analysis/training_checkpoints/",
    )
    return params


def training_command(config_directory, experiment_id):
    return [str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nproc_per_node=2",
            str(ROOT / "run_expid.py"), "--config", str(config_directory),
            "--expid", experiment_id, "--gpu", "0,1"]


def code_hashes():
    paths = [ROOT / "run_expid.py", ROOT / "pyproject.toml", ROOT / "uv.lock",
             Path(__file__).resolve()]
    for directory in ("unirank", "model_zoo"):
        paths.extend(sorted((ROOT / directory).rglob("*.py")))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare(task_id, attempt):
    task = TASKS[task_id]
    params = baseline_params(task, attempt)
    output = ARTIFACTS / "matched_baselines" / params["model_id"]
    config = output / "config"
    config.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML; an empty dataset override retains every frozen field.
    (config / "model_config.yaml").write_text(json.dumps(
        {"Base": {}, params["model_id"]: params}, indent=2) + "\n")
    (config / "dataset_config.yaml").write_text(json.dumps({task[1]: {}}, indent=2) + "\n")
    manifest = {
        "task_id": task_id, "attempt": attempt, "model": task[0], "dataset": task[1],
        "run_id": params["model_id"], "reference_log": str(reference_log(task).relative_to(ROOT)),
        "reference_log_sha256": hashlib.sha256(reference_log(task).read_bytes()).hexdigest(),
        "reference_params": reference_params(task), "effective_params": params,
        "allowed_differences": ["sisa_enabled", "model_id", "model_root"],
        "source_sha256": code_hashes(),
        "command": training_command(config, params["model_id"]),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_training_revision_verified": False,
        "scope": "New matched diagnostic at seed 20262028, not reconstruction of the 20262027 model instance.",
    }
    freeze = output / "frozen_manifest.json"
    if freeze.exists():
        previous = json.loads(freeze.read_text())
        for key in ("effective_params", "source_sha256", "reference_log_sha256"):
            if previous[key] != manifest[key]:
                raise RuntimeError(f"Frozen execution context changed: {key}")
        return output, previous
    write_json(freeze, manifest)
    return output, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, choices=range(len(TASKS)), required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    output, manifest = prepare(args.task_id, args.attempt)
    if args.prepare_only:
        print(json.dumps({"manifest": str(output / "frozen_manifest.json"),
                          "command": manifest["command"]}, indent=2), flush=True)
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Formal baseline training requires a Slurm allocation")
    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 2:
        raise RuntimeError("Exactly two allocated GPUs are required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) < 16:
        raise RuntimeError("At least 16 allocated CPUs required")
    if int(os.environ.get("SLURM_MEM_PER_NODE", 0)) < 480 * 1024:
        raise RuntimeError("At least 480 GiB host memory required")
    gpu_names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()
    if len(gpu_names) != 2 or any("H100" not in name for name in gpu_names):
        raise RuntimeError(f"Unexpected allocated GPU models: {gpu_names}")
    checkpoint_dir = ROOT / manifest["effective_params"]["model_root"] / manifest["dataset"]
    for suffix in (".model", ".log"):
        if (checkpoint_dir / (manifest["run_id"] + suffix)).exists():
            raise FileExistsError("Run output already exists; inspect the previous attempt before retrying")
    receipt = output / "started.json"
    descriptor = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"job_id": os.environ["SLURM_JOB_ID"], "gpus": gpu_names,
                   "started_at_utc": datetime.now(timezone.utc).isoformat()}, stream, indent=2)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
                       TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="3600")
    if manifest["dataset"] == "MerRec_Action":
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with tempfile.TemporaryDirectory(prefix="rise-matched-baseline-") as temporary:
        environment.update(TMPDIR=temporary, TMP=temporary, TEMP=temporary,
                           TORCHINDUCTOR_CACHE_DIR=f"{temporary}/torchinductor",
                           TRITON_CACHE_DIR=f"{temporary}/triton")
        completed = subprocess.run(manifest["command"], cwd=ROOT, env=environment, check=False)
    write_json(output / "outcome.json", {
        "returncode": completed.returncode, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"], "metric_audit_pending": True,
    })
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
