"""Freeze and launch exactly three targeted controls against historical E2028.

Only model_id differs in the effective Params. The treatment is a documented
process-local adapter-construction patch; no repository model code is modified.
--dry-run is read-only. --prepare-only writes the isolated reviewable manifest.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent
for directory in (ROOT, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from targeted_controls import audit_model, patched_build

ARTIFACTS = ROOT / "artifacts/rise_targeted_controls"
TASKS = (
    {"model": "HyFormer", "dataset": "MerRec_Action", "mode": "fixed_cumulative", "reference_task": "548166_1", "expected_dense": 7_804_797},
    {"model": "HiFormer", "dataset": "QK_Video_Action", "mode": "target_only", "reference_task": "548166_10", "expected_dense": 27_826_102},
    {"model": "HiFormer", "dataset": "QK_Video_Action", "mode": "field_only", "reference_task": "548166_10", "expected_dense": 27_887_120},
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_log(task):
    return ROOT / "checkpoints" / task["dataset"] / (
        f"SISA_single_seed_{task['model']}_{task['dataset']}_sisa_seed20262028_ws2_bs16384_acc1_attempt1.log")


def reference_params(task):
    text = reference_log(task).read_text()
    if text.count("INFO Params:") != 1 or text.count("Test evaluation") != 1:
        raise ValueError("Historical reference must have a unique complete run")
    raw, _ = json.JSONDecoder().raw_decode(text.split("INFO Params:", 1)[1].lstrip())
    params = {}
    for key, value in raw.items():
        try:
            params[key] = ast.literal_eval(value) if isinstance(value, str) else value
        except (ValueError, SyntaxError):
            params[key] = value
    required = {"model": task["model"], "dataset_id": task["dataset"], "seed": 20262028,
                "world_size": 2, "batch_size": 16384, "epochs": 1, "max_len": 100,
                "dataloader_seed": 2027, "sisa_parameter_seed": 20260822,
                "sisa_enabled": True, "sisa_score_dim": 16, "sisa_lambda_init": .1,
                "sisa_score_scale": 1., "enable_bf16": True, "enable_torch_compile": True,
                "num_layers": 3, "num_heads": 2, "token_dim": 256, "embedding_dim": 16}
    changed = [key for key, value in required.items() if params.get(key) != value]
    if changed:
        raise ValueError(f"Historical targeted protocol changed: {changed}")
    return params


def identifier(task_id, attempt):
    if task_id not in range(len(TASKS)) or attempt < 1:
        raise ValueError("Invalid task or attempt")
    task = TASKS[task_id]
    return f"RISE_targeted_{task['model']}_{task['dataset']}_{task['mode']}_seed20262028_attempt{attempt}"


def effective_params(task_id, attempt):
    task = TASKS[task_id]
    return dict(reference_params(task), model_id=identifier(task_id, attempt))


def full_site_parameters(hidden, heads, dimension):
    return heads * hidden * (2 * dimension + 1 + dimension // 2) + 2 * heads


def expected_counts(task_id):
    task = TASKS[task_id]
    params = reference_params(task)
    text = reference_log(task).read_text()
    dense = int(re.findall(r"Number of dense parameters: (\d+)\.", text)[0])
    embedding = int(re.findall(r"Number of embedding parameters: (\d+)\.", text)[0])
    total = int(re.findall(r"Total number of parameters: (\d+)\.", text)[0])
    if total != dense + embedding:
        raise ValueError("Historical parameter arithmetic is inconsistent")
    layers, heads, width, dimension = params["num_layers"], params["num_heads"], params["token_dim"], params["sisa_score_dim"]
    if task["mode"] == "fixed_cumulative":
        sites = 2 * layers
        removed = sites * (heads * width + heads * (dimension // 2) * width)
        added = sites * heads * (dimension // 2)
        active_dense, stored = dense - removed + added, total + added
        detail = {"sites": sites, "frozen_input_control_parameters": removed,
                  "added_learnable_constant_phases": added, "parameter_matched": False,
                  "fixed_initialization": "Original per-head decay bias -5; phase increments zero, as RankMixer fixed control"}
    else:
        item_width = sum(int(column.get("embedding_dim", params["embedding_dim"])) *
                         (len(column["name"]) if isinstance(column["name"], list) else 1)
                         for column in params["feature_cols"] if column.get("active", True)
                         and column.get("type") != "meta" and column.get("source") in {"item", "action"})
        target = full_site_parameters(item_width, 1, dimension)
        fields = layers * full_site_parameters(width, heads, dimension)
        removed = fields if task["mode"] == "target_only" else target
        active_dense, stored = dense - removed, total - removed
        detail = {"target_site_parameters": target, "three_field_site_parameters": fields,
                  "target_hidden_size": item_width, "target_heads": 1,
                  "field_hidden_size": width, "field_heads": heads,
                  "removed_site_parameters": removed, "disabled_sites": "None; no inactive tensors retained"}
    if active_dense != task["expected_dense"]:
        raise ValueError("Expected dense count differs from the fixed study design")
    return {"historical_E_active_dense": dense, "active_dense": active_dense,
            "active_embedding": embedding, "active_total": embedding + active_dense,
            "stored_total": stored, "derivation": detail}


def source_hashes():
    paths = [Path(__file__), SCRIPTS / "targeted_controls.py", SCRIPTS / "submit_targeted_controls.sbatch",
             ROOT / "run_expid.py", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    for directory in ("unirank", "model_zoo"):
        paths.extend(sorted((ROOT / directory).rglob("*.py")))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def training_command(task_id, attempt):
    return [str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nproc_per_node=2",
            str(Path(__file__).resolve()), "--worker", "--task-id", str(task_id), "--attempt", str(attempt)]


def review_manifest(task_id, attempt):
    task = TASKS[task_id]
    params = effective_params(task_id, attempt)
    reference = reference_params(task)
    differences = [key for key in params.keys() | reference.keys() if params.get(key) != reference.get(key)]
    if differences != ["model_id"]:
        raise ValueError(f"Unexpected effective parameter differences: {differences}")
    return {"study": "rise_targeted_controls", "task_id": task_id, "attempt": attempt, **task,
            "run_id": params["model_id"], "reference_job_task": task["reference_task"],
            "reference_log": str(reference_log(task).relative_to(ROOT)),
            "reference_log_sha256": sha256(reference_log(task)), "reference_params": reference,
            "effective_params": params, "allowed_parameter_differences": ["model_id"],
            "process_local_treatment": task["mode"], "parameter_expectations": expected_counts(task_id),
            "source_sha256": source_hashes(), "command": training_command(task_id, attempt),
            "checkpoint_restore_requires": "Construct under targeted_controls.patched_build(mode), then strict state_dict loading",
            "scope": "New seed20262028 targeted training; not restoration of seed20262027; fixed control is not parameter-matched",
            "historical_training_source_revision_verified": False}


def write_json(path, value):
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def prepare(task_id, attempt):
    manifest = review_manifest(task_id, attempt)
    output = ARTIFACTS / manifest["run_id"]
    config = output / "config"
    config.mkdir(parents=True, exist_ok=True)
    freeze = output / "frozen_manifest.json"
    if freeze.exists():
        previous = json.loads(freeze.read_text())
        if {key: value for key, value in previous.items() if key != "frozen_at_utc"} != manifest:
            raise ValueError("Frozen targeted execution context changed; do not overwrite an attempt")
        manifest = previous
    else:
        manifest["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(freeze, manifest)
    configs = {"model_config.yaml": {"Base": {}, manifest["run_id"]: manifest["effective_params"]},
               "dataset_config.yaml": {manifest["dataset"]: {}}}
    for name, value in configs.items():
        path = config / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("Frozen runtime configuration changed")
        if not path.exists():
            write_json(path, value)
    return output, manifest


@contextmanager
def audited_construction(model_class, mode, expected_dense, output, rank):
    """Audit-only constructor wrapper; adds no parameters or random draws."""
    original = model_class.__init__

    def checked(self, *args, **kwargs):
        original(self, *args, **kwargs)
        audit = audit_model(self, mode, expected_dense)
        audit.update(rank=rank, model_id=self.model_id)
        write_json(output / f"model_audit_rank{rank}.json", audit)
        print("RISE_TARGETED_MODEL_AUDIT " + json.dumps(audit), flush=True)

    model_class.__init__ = checked
    try:
        yield
    finally:
        model_class.__init__ = original


def run_worker(task_id, attempt):
    import model_zoo
    if int(os.environ.get("WORLD_SIZE", 0)) != 2:
        raise RuntimeError("Targeted worker requires exactly two ranks")
    output = ARTIFACTS / identifier(task_id, attempt)
    manifest = json.loads((output / "frozen_manifest.json").read_text())
    if {key: value for key, value in manifest.items() if key != "frozen_at_utc"} != review_manifest(task_id, attempt):
        raise RuntimeError("Worker source/config differs from the frozen plan")
    task = TASKS[task_id]
    rank = int(os.environ.get("RANK", -1))
    print("RISE_TARGETED_WORKER " + json.dumps({"task_id": task_id, "mode": task["mode"], "rank": rank,
                                               "run_id": manifest["run_id"]}), flush=True)
    sys.argv = [str(ROOT / "run_expid.py"), "--config", str(output / "config"),
                "--expid", manifest["run_id"], "--gpu", "0,1"]
    with patched_build(task["mode"]), audited_construction(
            getattr(model_zoo, task["model"]), task["mode"], task["expected_dense"], output, rank):
        runpy.run_path(str(ROOT / "run_expid.py"), run_name="__main__")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, choices=range(3), required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error("Attempt must be positive")
    os.chdir(ROOT)
    if args.dry_run:
        print(json.dumps(review_manifest(args.task_id, args.attempt), indent=2))
        return 0
    if args.worker:
        return run_worker(args.task_id, args.attempt)
    output, manifest = prepare(args.task_id, args.attempt)
    if args.prepare_only:
        print(json.dumps({"frozen_manifest": str(output / "frozen_manifest.json"),
                          "run_id": manifest["run_id"], "counts": manifest["parameter_expectations"],
                          "command": manifest["command"]}, indent=2))
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Formal targeted training requires a Slurm allocation")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 2 or any(not value for value in visible):
        raise RuntimeError("Exactly two allocated GPUs required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) < 16 or int(os.environ.get("SLURM_MEM_PER_NODE", 0)) < 480 * 1024:
        raise RuntimeError("16 CPUs and 480 GiB host memory required")
    names = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()
    if len(names) != 2 or any("H100" not in name for name in names):
        raise RuntimeError(f"Expected two H100 GPUs, got {names}")
    for suffix in (".model", ".log"):
        path = ROOT / "checkpoints" / manifest["dataset"] / (manifest["run_id"] + suffix)
        if path.exists():
            raise FileExistsError(f"Attempt output exists: {path}; use a new reviewed attempt")
    started_path = output / "started.json"
    with started_path.open("x") as stream:
        json.dump({"job_id": os.environ["SLURM_JOB_ID"], "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                   "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "gpu_names": names,
                   "started_utc": datetime.now(timezone.utc).isoformat()}, stream, indent=2)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="3600")
    if manifest["dataset"] == "MerRec_Action":
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with tempfile.TemporaryDirectory(prefix="rise-targeted-control-") as temporary:
        environment.update(TMPDIR=temporary, TMP=temporary, TEMP=temporary,
                           TORCHINDUCTOR_CACHE_DIR=f"{temporary}/torchinductor", TRITON_CACHE_DIR=f"{temporary}/triton")
        result = subprocess.run(manifest["command"], cwd=ROOT, env=environment, check=False)
    write_json(output / "outcome.json", {"returncode": result.returncode, "job_id": os.environ["SLURM_JOB_ID"],
                                        "finished_utc": datetime.now(timezone.utc).isoformat()})
    if result.returncode == 0:
        print(f"RISE_TARGETED_COMPLETE run_id={manifest['run_id']}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
