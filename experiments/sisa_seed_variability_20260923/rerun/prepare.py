"""Freeze the eight selected low-result retries without changing training code."""

import ast
import csv
import hashlib
import importlib.metadata
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import run_onetrans_seed_completion as one4
from scripts import run_onetrans_seed_completion_ws2 as one2
from scripts.sisa_single_seed_tasks import single_seed_task
from unirank.utils import load_config

OUT = Path(__file__).parent
SELECTION = [
    ("Zenith", "TencentGR_10M_Action", 20262027, "macro_auc", 4, 12, 2),
    ("HyFormer", "TencentGR_10M_Action", 20262027, "macro_auc", 4, 13, 2),
    ("HyFormer", "MerRec_Action", 20262028, "macro_auc; Offer", 2, 1, 2),
    ("Zenith", "MerRec_Action", 20262029, "macro_auc; Purchase", 2, 0, 3),
    ("HiFormer", "MerRec_Action", 20262029, "Checkout", 2, 2, 2),
    ("HyFormer", "KuaiRand_Video_Action", 20262029, "is_follow", 2, 5, 2),
    ("OneTrans", "MerRec_Action", 20262028, "macro_auc", 4, 8, 2),
    ("OneTrans", "TencentGR_10M_Action", 20262029, "macro_logloss", 2, 5, 2),
]


def parse_value(value):
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


audit = list(csv.DictReader((OUT.parent / "results/run_log_audit.csv").open()))
reference = {(r["model"], r["dataset"], int(r["seed"])): r for r in audit}
tasks = []
checks = []
groups = {}
for model, dataset, seed, reason, world, task_id, attempt in SELECTION:
    original = reference[model, dataset, seed]
    assert digest(ROOT / original["checkpoint_log"]) == original["checkpoint_log_sha256"]
    text = (ROOT / original["checkpoint_log"]).read_text()
    raw, _ = json.JSONDecoder().raw_decode(text.split("Params: ", 1)[1])
    old = {k: parse_value(v) for k, v in raw.items()}
    batch = 32768 // world
    if model == "OneTrans":
        runner = one4 if world == 4 else one2
        task = runner.study_task(task_id, attempt)
        assert task.dataset == dataset and task.seed == seed
        current = load_config(str(one4.CONFIG_DIR), task.experiment)
        runner.validate_protocol(task, current)
        runner.ensure_fresh_outputs(task)
        effective = runner.effective_params(task, current)
        run_id = task.run_id
        command = runner.training_command(task)
        env = {"ONETRANS_ATTEMPT" if world == 4 else "ONETRANS_WS2_ATTEMPT": str(attempt)}
        launcher = "scripts/submit_onetrans_seed_completion" + ("" if world == 4 else "_ws2") + ".sbatch"
        group = f"one-ws{world}"
    else:
        mapping = single_seed_task(task_id)
        assert (mapping.model, mapping.dataset) == (model, dataset)
        protocol = f"ws{world}_bs{batch}_acc1"
        run_id = f"SISA_single_seed_{model}_{dataset}_sisa_seed{seed}_{protocol}_attempt{attempt}"
        current = load_config("./config", mapping.experiment)
        effective = dict(current, seed=seed, dataloader_seed=seed - 20260001,
                         sisa_parameter_seed=seed - 1206, batch_size=batch,
                         sisa_enabled=True, sisa_score_dim=16, sisa_lambda_init=0.1,
                         sisa_score_scale=1.0, enable_bf16=True, model_id=run_id,
                         world_size=world, distributed=True, rank=0, local_rank=0, gpu=0)
        if dataset == "MerRec_Action":
            effective.update(sparse_optimizer_foreach=False, sparse_adagrad_chunk_size=16777216)
        command = ["bash", "scripts/run_sisa_single_seed_task.sh"]
        env = {"SISA_WORLD_SIZE": str(world), "SISA_BATCH_SIZE": str(batch),
               "SISA_PROTOCOL": protocol, "SISA_SEED": str(seed),
               "SISA_DATALOADER_SEED": str(seed - 20260001),
               "SISA_PARAMETER_SEED": str(seed - 1206),
               "SISA_STUDY": f"sisa_single_seed{seed}", "SISA_ATTEMPT": str(attempt)}
        launcher = "scripts/run_sisa_single_seed_task.sh"
        group = f"s{str(seed)[-2:]}-a{attempt}"
        (ROOT / "artifacts" / env["SISA_STUDY"] / "telemetry").mkdir(parents=True, exist_ok=True)
    # Historical seed27 omitted these keys; these are its documented defaults.
    old.setdefault("dataloader_seed", 2026)
    old.setdefault("sisa_parameter_seed", 20260821)
    old["model_id"] = run_id
    differences = {k: [old.get(k), effective.get(k)] for k in old.keys() | effective.keys()
                   if old.get(k) != effective.get(k)}
    assert not differences, (model, dataset, seed, differences)
    for suffix in (".log", ".model"):
        assert not (ROOT / "checkpoints" / dataset / (run_id + suffix)).exists(), run_id
    tasks.append(dict(model=model, dataset=dataset, seed=seed, reason=reason,
                      original_job_task_id=original["job_task_id"],
                      original_checkpoint_log=original["checkpoint_log"],
                      run_id=run_id, group=group, task_id=task_id, attempt=attempt,
                      world_size=world, per_gpu_batch=batch, global_batch=32768,
                      training_command=command, effective_params=effective))
    checks.append(dict(model=model, dataset=dataset, seed=seed,
                       effective_parameter_differences=differences))
    g = groups.setdefault(group, dict(world=world, env=env, launcher=launcher, task_ids=[]))
    assert g["env"] == env
    g["task_ids"].append(task_id)

commands = []
verify = str((OUT / "verify_frozen.py").relative_to(ROOT))
for group, g in groups.items():
    wrap = shlex.join([".venv/bin/python", verify]) + " && " + shlex.join(["bash", g["launcher"]])
    commands.append(dict(group=group, task_ids=g["task_ids"], command=[
        "sbatch", "--parsable", "--partition=acd_u", f"--job-name=ur-lowretry-{group}",
        "--nodes=1", "--ntasks=1", f"--cpus-per-task={g['world'] * 8}",
        f"--mem={g['world'] * 240}G", f"--gres=gpu:{g['world']}", "--time=12:00:00",
        "--no-requeue", "--output=logs/%x-%A_%a.out",
        "--array=" + ",".join(str(i) for i in g["task_ids"]) + "%2",
        "--export=ALL," + ",".join(k + "=" + v for k, v in g["env"].items()),
        "--wrap=" + wrap,
    ]))
assert len(tasks) == 8 and sum(len(g["task_ids"]) for g in commands) == 8
source_paths = [ROOT / name for name in (
    "run_expid.py", "config/model_config.yaml", "config/dataset_config.yaml", "uv.lock", "pyproject.toml",
    "scripts/run_sisa_single_seed_task.sh", "scripts/sisa_single_seed_tasks.py",
    "scripts/run_onetrans_seed_completion.py", "scripts/run_onetrans_seed_completion_ws2.py",
    "scripts/submit_onetrans_seed_completion.sbatch", "scripts/submit_onetrans_seed_completion_ws2.sbatch",
)]
for directory in ("unirank", "model_zoo"):
    source_paths.extend((ROOT / directory).rglob("*.py"))
source_paths.extend(one4.CONFIG_DIR.glob("*.yaml"))
source_paths.extend(one4.STUDY_DIR.glob("configs/*.json"))
source_paths.append(OUT / "verify_frozen.py")
snapshot = dict(created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                git_branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(set(source_paths))},
                package_versions={p: importlib.metadata.version(p) for p in ("torch", "numpy", "polars")},
                configuration_checks=checks,
                historical_defaults={"dataloader_seed": 2026, "sisa_parameter_seed": 20260821})
(OUT / "preflight.json").write_text(json.dumps(snapshot, indent=2) + "\n")
(OUT / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
(OUT / "submission_commands.json").write_text(json.dumps(commands, indent=2) + "\n")
print(f"Prepared {len(tasks)} tasks in {len(commands)} submission groups; all effective parameters match.")
for task in tasks:
    print(task["group"], task["task_id"], task["model"], task["dataset"], task["seed"], task["attempt"])
