"""Collect this retry batch into a separate comparison without replacing originals."""

import argparse
import ast
import csv
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.collect_sisa_single_seed_results import parse_test_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--require-complete", action="store_true")
parser.add_argument("--batch-dir", type=Path, default=Path(__file__).parent)
args = parser.parse_args()
OUT = args.batch_dir.resolve()
jobs = {r["group"]: r["array_job_id"] for r in json.loads((OUT / "submission.json").read_text())}
tasks = json.loads((OUT / "tasks.json").read_text())
accounting = subprocess.run(
    ["sacct", "-n", "-X", "-j", ",".join(jobs.values()),
     "--format=JobID,State,ExitCode,NodeList,Elapsed", "--parsable2"],
    check=True, text=True, capture_output=True,
)
states = {}
for line in accounting.stdout.splitlines():
    fields = line.split("|")
    if len(fields) >= 5:
        states[fields[0]] = dict(state=fields[1], exit_code=fields[2], node=fields[3], elapsed=fields[4])
queue = subprocess.run(
    ["squeue", "-h", "-r", "-j", ",".join(jobs.values()), "-o", "%i|%T|%N|%M"],
    check=False, text=True, capture_output=True,
)
for line in queue.stdout.splitlines():
    fields = line.split("|")
    if len(fields) == 4 and fields[0] not in states:
        states[fields[0]] = dict(state=fields[1], exit_code="", node=fields[2], elapsed=fields[3])

results, labels = [], []
for task in tasks:
    job = f"{jobs[task['group']]}_{task['task_id']}"
    status = states.get(job, dict(state="PENDING_OR_UNAVAILABLE", exit_code="", node="", elapsed=""))
    slurm = ROOT / "logs" / f"ur-lowretry-{task['group']}-{job}.out"
    checkpoint = ROOT / "checkpoints" / task["dataset"] / (task["run_id"] + ".log")
    slurm_text = slurm.read_text(errors="replace") if slurm.exists() else ""
    checkpoint_text = checkpoint.read_text(errors="replace") if checkpoint.exists() else ""
    original = parse_test_metrics((ROOT / task["original_checkpoint_log"]).read_text())
    metrics = parse_test_metrics(checkpoint_text)
    metric_match = bool(metrics) and metrics == parse_test_metrics(slurm_text) and set(metrics) == set(original)
    marker = ("ONETRANS_SEED_COMPLETE" if task["world_size"] == 4 else "ONETRANS_WS2_COMPLETE") if task["model"] == "OneTrans" else "SISA_SINGLE_SEED_COMPLETE"
    params_match = False
    if "Params: " in checkpoint_text:
        raw, _ = json.JSONDecoder().raw_decode(checkpoint_text.split("Params: ", 1)[1])
        actual = {}
        for key, value in raw.items():
            try:
                actual[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                actual[key] = value
        params_match = actual == task["effective_params"]
    complete = status["state"] == "COMPLETED" and status["exit_code"] == "0:0" and marker in slurm_text and metric_match and params_match
    steps = re.findall(r"(\d+)/([\d]+)\s*\[", slurm_text)
    row = dict(model=task["model"], dataset=task["dataset"], seed=task["seed"],
               job_task_id=job, **status, complete=complete,
               parameter_match=params_match, final_metrics_match=metric_match,
               train_progress="/".join(steps[-1]) if steps else "",
               original_macro_auc=statistics.fmean(v["AUC"] for v in original.values()),
               new_macro_auc="", delta_macro_auc="",
               original_macro_logloss=statistics.fmean(v["logloss"] for v in original.values()),
               new_macro_logloss="", delta_macro_logloss="",
               checkpoint_log=str(checkpoint.relative_to(ROOT)), slurm_log=str(slurm.relative_to(ROOT)))
    if complete:
        for metric, field in (("auc", "AUC"), ("logloss", "logloss")):
            row[f"new_macro_{metric}"] = statistics.fmean(v[field] for v in metrics.values())
            row[f"delta_macro_{metric}"] = row[f"new_macro_{metric}"] - row[f"original_macro_{metric}"]
        for label in sorted(original):
            labels.append(dict(model=task["model"], dataset=task["dataset"], seed=task["seed"],
                               label=label, job_task_id=job, original_auc=original[label]["AUC"],
                               new_auc=metrics[label]["AUC"], delta_auc=metrics[label]["AUC"] - original[label]["AUC"],
                               original_logloss=original[label]["logloss"], new_logloss=metrics[label]["logloss"],
                               delta_logloss=metrics[label]["logloss"] - original[label]["logloss"]))
    results.append(row)

target = OUT / "results"
target.mkdir(exist_ok=True)
for name, rows in (("runs.csv", results), ("label_comparison.csv", labels)):
    if rows:
        with (target / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
snapshot = dict(checked_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                complete_count=sum(r["complete"] for r in results), expected_count=len(tasks), runs=results)
(target / "status.json").write_text(json.dumps(snapshot, indent=2) + "\n")
comparison_note = ("本批包含启动失败重提和新增种子实验，种子以表中记录为准；原始结果全部保留。"
                   if any(t.get("experiment_kind") == "new_seed" for t in tasks)
                   else "保留原结果，本表为相同种子的独立复跑对照；不按更高测试值替换原始三种子结果。")
lines = ["# 低值种子复跑结果", "", f"核验时间：{snapshot['checked_at']}。已通过最终测试核验 {snapshot['complete_count']}/{len(tasks)}。", "",
         comparison_note, "",
         "| 模型 | 数据集 | 种子 | 作业 | 状态 | 原 AUC | 新 AUC | 差值 |", "|---|---|---|---|---|---:|---:|---:|"]
for r in results:
    new = f"{r['new_macro_auc']:.7f}" if r["complete"] else "待完成核验"
    delta = f"{r['delta_macro_auc']:+.7f}" if r["complete"] else "—"
    lines.append(f"| {r['model']} | {r['dataset']} | {r['seed']} | {r['job_task_id']} | {r['state']} | {r['original_macro_auc']:.7f} | {new} | {delta} |")
(OUT / "report.md").write_text("\n".join(lines) + "\n")
print(json.dumps(dict(checked_at=snapshot["checked_at"], complete=snapshot["complete_count"],
                      runs=[{k: r[k] for k in ("job_task_id", "model", "dataset", "state", "node", "train_progress", "parameter_match")} for r in results]), indent=2))
if args.require_complete and snapshot["complete_count"] != len(tasks):
    raise SystemExit("Not all selected runs passed the completion audit")
