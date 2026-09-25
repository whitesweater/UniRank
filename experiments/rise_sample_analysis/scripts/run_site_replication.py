"""Matched cross-model placement replication on QK; six independent tasks.

All patches are process-local and leave existing experiments/core code intact.
Zenith target/token placement and HyFormer sequence/query placement are distinct
semantic comparisons; no field-attention label is assigned to HyFormer decoding.
"""

import argparse
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
for path in (ROOT, SCRIPTS):
    sys.path.insert(0, str(path))
from run_matched_baseline import reference_params
from unirank.pytorch.layers.attentions.sisa import SISAAttentionConfig

DATASET = "QK_Video_Action"
ARTIFACTS = ROOT / "artifacts/rise_site_replication"
TASKS = (("Zenith", "baseline"), ("HyFormer", "baseline"),
         ("Zenith", "history_only"), ("HyFormer", "sequence_only"),
         ("Zenith", "token_only"), ("HyFormer", "query_only"))
REFERENCES = {"history_only": "query", "token_only": "sequence_end",
              "sequence_only": "sequence_end", "query_only": "query"}


def identity(task, attempt):
    if task not in range(6) or attempt < 1:
        raise ValueError("Invalid task or attempt")
    model, mode = TASKS[task]
    return f"RISE_cross_sites_{model}_{DATASET}_{mode}_seed20262028_attempt{attempt}"


def params_for(task, attempt):
    model, mode = TASKS[task]
    params = reference_params((model, DATASET))
    params.update(model_id=identity(task, attempt), sisa_enabled=mode != "baseline")
    return params


@contextmanager
def placement(mode):
    if mode not in {"baseline", *REFERENCES}:
        raise ValueError("Unknown placement")
    original = SISAAttentionConfig.build

    def build(config, hidden_size, num_heads):
        if not config.enabled:
            return original(config, hidden_size, num_heads)
        if mode == "baseline":
            raise ValueError("Native baseline must disable SISA in frozen Params")
        if config.variant != "full":
            raise ValueError("Do not combine placement and component controls")
        if config.decay_reference == REFERENCES[mode]:
            return original(config, hidden_size, num_heads)
        return None

    SISAAttentionConfig.build = build
    try:
        yield
    finally:
        SISAAttentionConfig.build = original


def expected_sites(model):
    if model == "Zenith":
        return [("attention_layers", "query", 0, 80, 1)] + [
            (f"unified_interaction_layers.mixer_layers.{i}", "sequence_end", 100+i, 256, 2) for i in range(3)]
    if model == "HyFormer":
        return [(f"unified_interaction_layers.seq_layers.{i}", "sequence_end", i, 256, 2) for i in range(3)] + [
            (f"unified_interaction_layers.decode_layers.{i}", "query", 100+i, 256, 2) for i in range(3)]
    raise ValueError("Unsupported backbone")


def full_aux_count(width, heads):
    return heads * (41 * width + 2)


def expected_counts(task):
    model, mode = TASKS[task]
    log = ROOT / "checkpoints" / DATASET / f"SISA_single_seed_{model}_{DATASET}_sisa_seed20262028_ws2_bs16384_acc1_attempt1.log"
    raw = log.read_text()
    dense = int(re.findall(r"Number of dense parameters: (\d+)\.", raw)[0])
    embedding = int(re.findall(r"Number of embedding parameters: (\d+)\.", raw)[0])
    removed = sum(full_aux_count(width, heads) for _, reference, _, width, heads in expected_sites(model)
                  if mode == "baseline" or reference != REFERENCES[mode])
    return {"dense": dense - removed, "embedding": embedding, "total": dense - removed + embedding}


def unwrap(module):
    while hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


def audit_model(model, task, check_capacity=True):
    name, mode = TASKS[task]
    slots = []
    for path, reference, offset, width, heads in expected_sites(name):
        module = model
        for component in path.split("."):
            module = unwrap(module)
            module = module[int(component)] if component.isdigit() else getattr(module, component)
        bias = unwrap(module).sisa_score_bias
        enabled = mode != "baseline" and reference == REFERENCES[mode]
        if (bias is not None) != enabled:
            raise ValueError(f"Wrong placement: {path}")
        record = {"path": path, "reference": reference, "enabled": enabled}
        if enabled:
            if (bias.hidden_size, bias.num_heads, bias.score_dim, bias.parameter_seed) != (width, heads, 16, 20260822+offset):
                raise ValueError(f"Retained site identity changed: {path}")
            record.update(seed=bias.parameter_seed, hidden_size=width, heads=heads, score_dim=16)
        slots.append(record)
    dense = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "embedding" not in n)
    if check_capacity and dense != expected_counts(task)["dense"]:
        raise ValueError("Unexpected active dense capacity")
    return {"model": name, "mode": mode, "sites": slots, "active_sites": sum(s["enabled"] for s in slots), "dense": dense}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def manifest_for(task, attempt):
    model, mode = TASKS[task]
    reference = reference_params((model, DATASET))
    params = params_for(task, attempt)
    changed = sorted(k for k in params.keys() | reference.keys() if params.get(k) != reference.get(k))
    assert changed == (["model_id", "sisa_enabled"] if mode == "baseline" else ["model_id"])
    paths = [Path(__file__), SCRIPTS / "submit_site_replication.sbatch", SCRIPTS / "run_matched_baseline.py",
             ROOT / "run_expid.py", ROOT / "uv.lock", ROOT / "pyproject.toml"]
    for directory in ("unirank", "model_zoo"):
        paths += sorted((ROOT / directory).rglob("*.py"))
    return {"task": task, "attempt": attempt, "model": model, "dataset": DATASET, "mode": mode,
            "run_id": identity(task, attempt), "reference_params": reference, "effective_params": params,
            "reference_log": f"checkpoints/{DATASET}/SISA_single_seed_{model}_{DATASET}_sisa_seed20262028_ws2_bs16384_acc1_attempt1.log",
            "reference_log_sha256": sha(ROOT / "checkpoints" / DATASET / f"SISA_single_seed_{model}_{DATASET}_sisa_seed20262028_ws2_bs16384_acc1_attempt1.log"),
            "allowed_logged_differences": changed, "expected_counts": expected_counts(task),
            "expected_sites": expected_sites(model), "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
            "scope": "Cross-model placement replication; fixed QK protocol, not a new 2027 model instance.",
            "historical_training_source_revision_verified": False}


def prepare(task, attempt):
    manifest = manifest_for(task, attempt)
    # Normalize tuples before comparing serialized manifests.
    manifest = json.loads(json.dumps(manifest))
    output = ARTIFACTS / manifest["run_id"]
    config = output / "config"; config.mkdir(parents=True, exist_ok=True)
    path = output / "frozen_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError("Frozen replication context changed")
    else:
        write_json(path, manifest)
    for name, content in {"model_config.yaml": {"Base": {}, manifest["run_id"]: manifest["effective_params"]},
                          "dataset_config.yaml": {DATASET: {}}}.items():
        path = config / name
        if path.exists() and json.loads(path.read_text()) != content:
            raise RuntimeError("Frozen runtime configuration changed")
        if not path.exists():write_json(path, content)
    return output, manifest


def worker(task, attempt):
    import model_zoo
    output, manifest = prepare(task, attempt)
    rank = int(os.environ.get("RANK", -1))
    if int(os.environ.get("WORLD_SIZE", 0)) != 2:
        raise RuntimeError("Two training ranks required")
    cls = getattr(model_zoo, manifest["model"])
    original = cls.__init__
    def audited(self, *args, **kwargs):
        original(self, *args, **kwargs)
        report = audit_model(self, task)
        report.update(rank=rank, run_id=manifest["run_id"])
        write_json(output / f"model_audit_rank{rank}.json", report)
        print("RISE_SITE_REPLICATION_AUDIT " + json.dumps(report), flush=True)
    cls.__init__ = audited
    sys.argv = [str(ROOT / "run_expid.py"), "--config", str(output / "config"),
                "--expid", manifest["run_id"], "--gpu", "0,1"]
    try:
        with placement(manifest["mode"]):
            runpy.run_path(str(ROOT / "run_expid.py"), run_name="__main__")
    finally:
        cls.__init__ = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, choices=range(6), required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(); os.chdir(ROOT)
    output, manifest = prepare(args.task_id, args.attempt)
    if args.prepare_only:
        print(json.dumps({"task":args.task_id,"run_id":manifest["run_id"],"counts":manifest["expected_counts"]}),flush=True)
        return 0
    if args.worker:
        worker(args.task_id, args.attempt); return 0
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Formal training requires Slurm")
    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 2:
        raise RuntimeError("Exactly two allocated GPUs required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK",0)) < 16 or int(os.environ.get("SLURM_MEM_PER_NODE",0)) < 480*1024:
        raise RuntimeError("16 CPUs and 480 GiB required")
    names = subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).strip().splitlines()
    if len(names)!=2 or any("H100" not in name for name in names):raise RuntimeError(f"Expected 2 H100: {names}")
    for suffix in (".log",".model"):
        if (ROOT/"checkpoints"/DATASET/(manifest["run_id"]+suffix)).exists():
            raise FileExistsError("Existing run output; review before a new attempt")
    with (output/"started.json").open("x") as stream:
        json.dump({"job_id":os.environ["SLURM_JOB_ID"],"array_job_id":os.environ.get("SLURM_ARRAY_JOB_ID"),
                   "array_task_id":os.environ.get("SLURM_ARRAY_TASK_ID"),"gpu_names":names,
                   "started_utc":datetime.now(timezone.utc).isoformat()},stream,indent=2)
    command=[str(ROOT/".venv/bin/torchrun"),"--standalone","--nproc_per_node=2",str(Path(__file__).resolve()),
             "--worker","--task-id",str(args.task_id),"--attempt",str(args.attempt)]
    environment=dict(os.environ,PYTHONUNBUFFERED="1",OMP_NUM_THREADS="4",TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="3600")
    with tempfile.TemporaryDirectory(prefix="rise-site-replication-") as cache:
        environment.update(TMPDIR=cache,TMP=cache,TEMP=cache,TORCHINDUCTOR_CACHE_DIR=f"{cache}/inductor",TRITON_CACHE_DIR=f"{cache}/triton")
        result=subprocess.run(command,cwd=ROOT,env=environment,check=False)
    write_json(output/"outcome.json",{"returncode":result.returncode,"job_id":os.environ["SLURM_JOB_ID"],"finished_utc":datetime.now(timezone.utc).isoformat()})
    if result.returncode==0:print("RISE_SITE_REPLICATION_COMPLETE "+manifest["run_id"],flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
