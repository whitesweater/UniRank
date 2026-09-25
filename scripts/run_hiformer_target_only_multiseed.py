"""Complete eight HiFormer target-only RISE runs; reuse existing seed20262028.

Seed20262027 uses its recorded four-rank budget and reconstructed historical
subseed defaults. Seed20262029 uses its recorded two-rank budget and explicit
subseeds. The model treatment only removes the three field RISE modules.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import runpy
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ENTRY = Path(__file__).resolve()
SBATCH = ROOT / "scripts/submit_hiformer_target_only_multiseed.sbatch"
LEGACY_RUNNER = ROOT / "experiments/rise_sample_analysis/scripts/run_targeted_controls.py"
CONTROLS = LEGACY_RUNNER.with_name("targeted_controls.py")
REFERENCE_AUDIT = ROOT / "experiments/hiformer_target_only_multiseed/notes/reference_audit.json"
ARTIFACTS = ROOT / "artifacts/rise_hiformer_target_only_multiseed"
HISTORICAL_DATA_ROOT = "/data_nvme/user/ywhao/proj/UniRank/datasets"
DATA_PATH_KEYS = ("data_root",) + tuple(f"{split}_{suffix}" for split in ("train", "valid", "test")
                                      for suffix in ("data", "user_info", "item_info"))
DATASETS = ("QK_Video_Action", "MerRec_Action", "KuaiRand_Video_Action", "TencentGR_10M_Action")
DENSE_COUNTS = (27_826_102, 67_481_031, 97_014_120, 87_154_804)
REFERENCE_TASKS = {20262027: ("14811_9", "14810_15", "14850_11", "543257_3"),
                   20262029: ("549595_10", "549595_2", "549595_6", "549595_14")}


def make_task(seed, index):
    dataset = DATASETS[index]
    if seed == 20262027:
        prefix = "SISA_expansion" if dataset == "TencentGR_10M_Action" else "SISA_strict"
        name = f"{prefix}_HiFormer_{dataset}_sisa_seed{seed}.log"
        world, batch, loader_seed, parameter_seed = 4, 8192, 2026, 20260821
    else:
        name = f"SISA_single_seed_HiFormer_{dataset}_sisa_seed{seed}_ws2_bs16384_acc1_attempt1.log"
        world, batch, loader_seed, parameter_seed = 2, 16384, 2028, 20260823
    return {"model": "HiFormer", "dataset": dataset, "mode": "target_only", "seed": seed,
            "world_size": world, "batch_size": batch, "dataloader_seed": loader_seed,
            "sisa_parameter_seed": parameter_seed, "expected_dense": DENSE_COUNTS[index],
            "reference_task": REFERENCE_TASKS[seed][index],
            "reference_path": f"checkpoints/{dataset}/{name}"}


TASKS = tuple(make_task(seed, index) for seed in (20262027, 20262029) for index in range(4))
_SPEC = importlib.util.spec_from_file_location(f"{__name__}._audited_targeted_runner", LEGACY_RUNNER)
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)
sha256 = _BASE.sha256
write_json = _BASE.write_json
patched_build = _BASE.patched_build
audit_model = _BASE.audit_model


def identifier(task_id, attempt):
    if task_id not in range(len(TASKS)) or attempt < 1:
        raise ValueError("Invalid HiFormer multiseed task or attempt")
    task = TASKS[task_id]
    return (f"RISE_hiformer_target_only_{task['dataset']}_seed{task['seed']}"
            f"_ws{task['world_size']}_bs{task['batch_size']}_attempt{attempt}")


def reference_log(task):
    return ROOT / task["reference_path"]


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
    required = {"model": "HiFormer", "dataset_id": task["dataset"], "seed": task["seed"],
                "world_size": task["world_size"], "batch_size": task["batch_size"],
                "epochs": 1, "max_len": 100, "accumulation_steps": 1,
                "sisa_enabled": True, "sisa_score_dim": 16,
                "sisa_lambda_init": .1, "sisa_score_scale": 1., "enable_bf16": True,
                "enable_torch_compile": True, "num_layers": 3, "num_heads": 2,
                "token_dim": 256, "embedding_dim": 16}
    for key in ("dataloader_seed", "sisa_parameter_seed"):
        if task["seed"] != 20262027 or key in params:
            required[key] = task[key]
    changed = [key for key, value in required.items() if params.get(key) != value]
    if changed:
        raise ValueError(f"Historical multiseed protocol changed: {changed}")
    return params


def normalized_reference(task):
    params = reference_params(task).copy()
    defaults, relocations = {}, {}
    for key in ("dataloader_seed", "sisa_parameter_seed"):
        if key not in params:
            defaults[key] = task[key]
            params[key] = task[key]
    for key in DATA_PATH_KEYS:
        old = params[key]
        if old == HISTORICAL_DATA_ROOT or old.startswith(HISTORICAL_DATA_ROOT + "/"):
            params[key] = "./datasets" + old[len(HISTORICAL_DATA_ROOT):]
            relocations[key] = {"historical": old, "current": params[key]}
        if key == "data_root":
            expected = "./datasets"
        else:
            split, suffix = key.split("_", 1)
            expected = f"./datasets/{task['dataset']}/{split}/{suffix}"
        if params[key] != expected:
            raise ValueError(f"Unexpected historical dataset path: {key}={old}")
    return params, defaults, relocations


def effective_params(task_id, attempt):
    run_id = identifier(task_id, attempt)
    return dict(normalized_reference(TASKS[task_id])[0], model_id=run_id)


def expected_counts(task_id):
    # This private namespace only changes reference resolution and task metadata.
    return _BASE.expected_counts(task_id)


def source_hashes():
    paths = [ENTRY, SBATCH, LEGACY_RUNNER, CONTROLS, REFERENCE_AUDIT,
             ROOT / "experiments/sisa_seed_variability_20260923/rerun/prepare.py",
             ROOT / "run_expid.py", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    for directory in ("unirank", "model_zoo"):
        paths.extend(sorted((ROOT / directory).rglob("*.py")))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def training_command(task_id, attempt):
    identifier(task_id, attempt)
    return [str(ROOT / ".venv/bin/torchrun"), "--standalone",
            f"--nproc_per_node={TASKS[task_id]['world_size']}", str(ENTRY),
            "--worker", "--task-id", str(task_id), "--attempt", str(attempt)]


def seed_bundle(task):
    return {key: task[key] for key in ("seed", "dataloader_seed", "sisa_parameter_seed")}


def review_manifest(task_id, attempt):
    run_id = identifier(task_id, attempt)
    task = TASKS[task_id]
    raw = reference_params(task)
    normalized, defaults, relocations = normalized_reference(task)
    params = effective_params(task_id, attempt)
    differences = sorted(key for key in params.keys() | raw.keys() if params.get(key) != raw.get(key))
    allowed = sorted({"model_id"} | defaults.keys() | relocations.keys())
    if differences != allowed:
        raise ValueError(f"Unexpected effective parameter differences: {differences}")
    return {"study": "rise_hiformer_target_only_multiseed", "task_id": task_id,
            "attempt": attempt, **task, "run_id": run_id, "seed_bundle": seed_bundle(task),
            "global_batch": task["world_size"] * task["batch_size"],
            "reference_job_task": task["reference_task"],
            "reference_log": str(reference_log(task).relative_to(ROOT)),
            "reference_log_sha256": sha256(reference_log(task)), "reference_params": raw,
            "normalized_reference_params": normalized, "effective_params": params,
            "allowed_parameter_differences": allowed,
            "historical_default_additions": defaults, "path_relocation_map": relocations,
            "process_local_treatment": "target_only", "parameter_expectations": expected_counts(task_id),
            "construction_contract": {"enabled_target_sites": 1, "enabled_field_sites": 0,
                                      "target_parameter_seed": task["sisa_parameter_seed"],
                                      "disabled_sites": "None; no inactive field adapter tensors"},
            "source_sha256": source_hashes(), "command": training_command(task_id, attempt),
            "historical_training_source_revision_verified": False,
            "historical_seed_provenance": (
                "Seed20262027 omitted subseeds; reconstructed dataloader=2026 and adapter=20260821 "
                "from historical defaults and rerun/prepare.py, with evidence in reference_audit.json. "
                "The historical training source revision and dataset byte identity cannot be fully recovered."
                if defaults else "Subseeds explicitly recorded in the historical training log."),
            "checkpoint_restore_requires": "Construct under targeted_controls.patched_build('target_only'), then strict state_dict loading",
            "scope": "Eight new runs plus four reused seed20262028 runs; descriptive three-seed evidence. "
                     "World size and drop_last sample coverage differ across seeds; not strict paired significance."}


def expected_configs(manifest):
    return {"model_config.yaml": {"Base": {}, manifest["run_id"]: manifest["effective_params"]},
            "dataset_config.yaml": {manifest["dataset"]: {}}}


def verify_frozen(output, task_id, attempt):
    manifest = json.loads((output / "frozen_manifest.json").read_text())
    if {key: value for key, value in manifest.items() if key != "frozen_at_utc"} != review_manifest(task_id, attempt):
        raise ValueError("Frozen execution context changed; do not overwrite an attempt")
    for name, value in expected_configs(manifest).items():
        if json.loads((output / "config" / name).read_text()) != value:
            raise ValueError("Frozen runtime configuration changed")
    for relative, digest in manifest["source_sha256"].items():
        if sha256(output / "source" / relative) != digest:
            raise ValueError(f"Frozen source snapshot changed: {relative}")
    return manifest


def prepare(task_id, attempt):
    manifest = review_manifest(task_id, attempt)
    output = ARTIFACTS / manifest["run_id"]
    if (output / "frozen_manifest.json").exists():
        return output, verify_frozen(output, task_id, attempt)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Incomplete attempt artifacts exist; use a new attempt")
    (output / "config").mkdir(parents=True, exist_ok=True)
    for name, value in expected_configs(manifest).items():
        write_json(output / "config" / name, value)
    for relative, digest in manifest["source_sha256"].items():
        source = (ROOT / relative).read_bytes()
        if hashlib.sha256(source).hexdigest() != digest:
            raise ValueError(f"Source changed during freeze: {relative}")
        path = output / "source" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source)
    manifest["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "frozen_manifest.json", manifest)
    return output, manifest


def rng_fingerprints():
    import numpy as np
    import torch
    values = {"torch_cpu": torch.get_rng_state().numpy().tobytes(),
              "python": pickle.dumps(random.getstate()), "numpy": pickle.dumps(np.random.get_state())}
    if torch.cuda.is_initialized():
        values["torch_cuda_current"] = torch.cuda.get_rng_state().cpu().numpy().tobytes()
    return {key: hashlib.sha256(value).hexdigest() for key, value in values.items()}


@contextmanager
def audited_construction(model_class, task, output, rank, run_id):
    original = model_class.__init__

    def checked(self, *args, **kwargs):
        original(self, *args, **kwargs)
        before = rng_fingerprints()
        audit = audit_model(self, "target_only", task["expected_dense"])
        if audit["sites"][0]["parameter_seed"] != task["sisa_parameter_seed"]:
            raise ValueError("Constructed target RISE uses the wrong parameter seed")
        after = rng_fingerprints()
        if before != after:
            raise ValueError("Construction audit changed random generator states")
        common = {"rank": rank, "world_size": task["world_size"], "run_id": run_id,
                  "seed_bundle": seed_bundle(task)}
        audit.update(**common, model_id=self.model_id)
        rng_audit = {**common, "before_audit": before, "after_audit": after, "audit_preserved_rng": True}
        write_json(output / f"model_audit_rank{rank}.json", audit)
        write_json(output / f"rng_audit_rank{rank}.json", rng_audit)
        print("RISE_TARGETED_MODEL_AUDIT " + json.dumps(audit), flush=True)
        print("RISE_TARGETED_RNG_AUDIT " + json.dumps(rng_audit), flush=True)

    model_class.__init__ = checked
    try:
        yield
    finally:
        model_class.__init__ = original


def gpu_argument(task):
    return ",".join(str(index) for index in range(task["world_size"]))


def run_worker(task_id, attempt):
    identifier(task_id, attempt)
    task = TASKS[task_id]
    world, rank = int(os.environ.get("WORLD_SIZE", 0)), int(os.environ.get("RANK", -1))
    if world != task["world_size"] or rank not in range(world):
        raise RuntimeError(f"Target-only worker requires exactly {task['world_size']} ranks")
    import model_zoo
    output = ARTIFACTS / identifier(task_id, attempt)
    manifest = verify_frozen(output, task_id, attempt)
    worker = {"task_id": task_id, "mode": "target_only", "rank": rank,
              "world_size": world, "run_id": manifest["run_id"], "seed_bundle": seed_bundle(task)}
    write_json(output / f"worker_rank{rank}.json", worker)
    print("RISE_TARGETED_WORKER " + json.dumps(worker), flush=True)
    sys.argv = [str(ROOT / "run_expid.py"), "--config", str(output / "config"),
                "--expid", manifest["run_id"], "--gpu", gpu_argument(task)]
    with patched_build("target_only"), audited_construction(
            getattr(model_zoo, "HiFormer"), task, output, rank, manifest["run_id"]):
        runpy.run_path(str(ROOT / "run_expid.py"), run_name="__main__")
    return 0


def validate_resources(task):
    world = task["world_size"]
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Formal target-only training requires a Slurm allocation")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != world or any(not value for value in visible):
        raise RuntimeError(f"Exactly {world} allocated GPUs required")
    if (int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) < 8 * world or
            int(os.environ.get("SLURM_MEM_PER_NODE", 0)) < 240 * world * 1024):
        raise RuntimeError(f"{8 * world} CPUs and {240 * world} GiB host memory required")
    names = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()
    if len(names) != world or len(set(names)) != 1:
        raise RuntimeError(f"Expected {world} visible GPUs of one model, got {names}")
    return names


def ensure_fresh_outputs(output, task, run_id):
    for suffix in (".model", ".log"):
        path = ROOT / "checkpoints" / task["dataset"] / (run_id + suffix)
        if path.exists():
            raise FileExistsError(f"Attempt output exists: {path}; use a new attempt")
    for name in ("started.json", "outcome.json"):
        if (output / name).exists():
            raise FileExistsError(f"Attempt already started: {output / name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, choices=range(len(TASKS)), required=True)
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
                          "run_id": manifest["run_id"], "command": manifest["command"]}, indent=2))
        return 0
    task = TASKS[args.task_id]
    names = validate_resources(task)
    ensure_fresh_outputs(output, task, manifest["run_id"])
    with (output / "started.json").open("x") as stream:
        json.dump({"job_id": os.environ["SLURM_JOB_ID"], "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                   "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "gpu_names": names,
                   "world_size": task["world_size"], "seed_bundle": seed_bundle(task),
                   "started_utc": datetime.now(timezone.utc).isoformat()}, stream, indent=2)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="3600")
    if task["dataset"] == "MerRec_Action":
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    try:
        with tempfile.TemporaryDirectory(prefix="rise-hiformer-multiseed-") as temporary:
            environment.update(TMPDIR=temporary, TMP=temporary, TEMP=temporary,
                               TORCHINDUCTOR_CACHE_DIR=f"{temporary}/torchinductor", TRITON_CACHE_DIR=f"{temporary}/triton")
            result = subprocess.run(manifest["command"], cwd=ROOT, env=environment, check=False)
        code, error = result.returncode, None
    except Exception as exc:
        code, error = 1, f"{type(exc).__name__}: {exc}"
    write_json(output / "outcome.json", {"returncode": code, "error": error,
                                        "job_id": os.environ["SLURM_JOB_ID"],
                                        "finished_utc": datetime.now(timezone.utc).isoformat()})
    if code == 0:
        print(f"RISE_TARGETED_COMPLETE run_id={manifest['run_id']}", flush=True)
    elif error:
        print(error, file=sys.stderr, flush=True)
    return code


_BASE.TASKS = TASKS
_BASE.reference_log = lambda task: reference_log(task)
_BASE.reference_params = lambda task: normalized_reference(task)[0]


if __name__ == "__main__":
    raise SystemExit(main())
