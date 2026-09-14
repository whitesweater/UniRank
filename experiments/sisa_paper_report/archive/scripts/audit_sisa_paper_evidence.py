#!/usr/bin/env python3
"""Reconstruct the accepted SISA paper evidence from archived metrics and logs.

No training imports, no source archive mutation, and no full dataset scan.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


MODELS = ("HiFormer", "HyFormer", "RankMixer", "Zenith")
LABELS = {
    "QK_Video_Action": ("click", "follow", "like", "share"),
    "KuaiRand_Video_Action": ("is_click", "is_follow", "is_like", "is_comment", "is_forward", "long_view"),
    "TencentGR_10M_Action": ("is_click", "is_conversion"),
    "MerRec_Action": ("Like", "Cart", "Offer", "Checkout", "Purchase"),
}
NAMES = dict(zip(LABELS, ("QK-Video", "KuaiRand", "TAAC-25", "MerRec")))
SEEDS = (20262027, 20262028, 20262029)
STUDIES = ("sisa_native_strict", "sisa_expansion_acd", "sisa_single_seed20262028", "sisa_single_seed20262029")
TRAIN_FIELDS = (
    "batch_size", "accumulation_steps", "epochs", "seed", "max_len", "embedding_dim",
    "num_layers", "token_dim", "num_heads", "dense_optimizer", "sparse_optimizer",
    "dense_learning_rate", "sparse_learning_rate", "dense_weight_decay", "enable_bf16",
    "enable_torch_compile", "gradient_checkpointing", "blocked", "group_id", "loss",
    "monitor", "monitor_mode", "eval_steps", "feature_cols", "label_col",
    "train_data", "valid_data", "test_data", "sisa_enabled", "sisa_score_dim",
    "sisa_lambda_init", "sisa_score_scale", "sisa_decay_bias", "sisa_exp_clamp",
    "sisa_parameter_seed", "dataloader_seed", "sparse_optimizer_foreach",
    "sparse_adagrad_chunk_size", "num_ns_token", "num_global_token", "num_token", "id_dim", "qkv_rank",
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def accepted(row):
    return row["model"] in MODELS and row["dataset"] in LABELS


def key(row):
    return row["model"], row["dataset"], row["label"]


def validate_keys(rows):
    actual = [key(row) for row in rows]
    expected = {(m, d, label) for m in MODELS for d, labels in LABELS.items() for label in labels}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Expected exactly the accepted 68 unique model/dataset/label keys")


def parse_test_metrics(text):
    if "Test evaluation" not in text:
        raise ValueError("No explicit final test evaluation boundary")
    tail = text.rsplit("Test evaluation", 1)[1]
    metrics = {}
    for label, values in re.findall(r"\[Task: ([^\]]+)\]\[Metrics\] ([^\n]+)", tail):
        if label in metrics:
            raise ValueError(f"Duplicate final test metric for {label}")
        parsed = {name: float(value) for name, value in re.findall(r"(AUC|logloss): ([0-9.eE+\-]+)", values)}
        if set(parsed) != {"AUC", "logloss"} or not all(math.isfinite(x) for x in parsed.values()):
            raise ValueError(f"Invalid final test metrics for {label}")
        if not 0 <= parsed["AUC"] <= 1 or parsed["logloss"] < 0:
            raise ValueError(f"Out-of-range metrics for {label}")
        metrics[label] = parsed
    if not metrics:
        raise ValueError("No final test metrics")
    return metrics


def summarize(rows, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["dataset"]].append(row)
    auc = [float(r[f"sisa_{seed}_auc"]) - float(r["baseline_auc"]) for r in rows]
    loss = [float(r[f"sisa_{seed}_logloss"]) - float(r["baseline_logloss"]) for r in rows]
    cell_auc = [statistics.fmean(float(r[f"sisa_{seed}_auc"]) - float(r["baseline_auc"]) for r in group) for group in groups.values()]
    cell_loss = [statistics.fmean(float(r[f"sisa_{seed}_logloss"]) - float(r["baseline_logloss"]) for r in group) for group in groups.values()]
    return {
        "seed": seed, "cells": len(groups), "labels": len(rows),
        "cell_macro_delta_auc": statistics.fmean(cell_auc),
        "cell_macro_delta_logloss": statistics.fmean(cell_loss),
        "label_macro_delta_auc": statistics.fmean(auc),
        "label_macro_delta_logloss": statistics.fmean(loss),
        "positive_auc_cells": sum(x > 1e-12 for x in cell_auc),
        "lower_logloss_cells": sum(x < -1e-12 for x in cell_loss),
        "positive_auc_labels": sum(x > 1e-12 for x in auc),
        "lower_logloss_labels": sum(x < -1e-12 for x in loss),
    }


def build(repository, output):
    import pyarrow.parquet as pq

    result_dir = output / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def tracked(path):
        content = path.read_bytes()
        manifest.append({"path": path.relative_to(repository).as_posix(), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        return content.decode("utf-8", errors="replace")

    runs, metrics, params, coverage = [], {}, {}, []
    for study in STUDIES:
        run_file = repository / "experiments" / study / "results/runs.csv"
        metric_file = run_file.with_name("metrics.csv")
        tracked(run_file)
        tracked(metric_file)
        archived = {(r["model"], r["dataset"], r["setting"], r["label"]): r for r in read_csv(metric_file)}
        for run in filter(accepted, read_csv(run_file)):
            if run["complete"] != "True" or run["protocol_valid"] != "True" or int(run["error_count"]) != 0:
                raise ValueError(f"Incomplete run: {run['job_task_id']}")
            seed = int(run.get("seed", 20262027))
            ws = int(run.get("world_size", 4))
            batch = int(run.get("per_gpu_batch", 8192))
            metric_text = tracked(repository / run["checkpoint_log"])
            slurm_text = tracked(repository / run["slurm_log"])
            parsed = parse_test_metrics(metric_text)
            if parsed != json.loads(run["metrics_json"]):
                raise ValueError(f"Run JSON differs from original final test: {run['job_task_id']}")
            if set(parsed) != set(LABELS[run["dataset"]]):
                raise ValueError("Label mapping differs")
            param_match = re.findall(r"INFO Params: (\{.*?\n\})", metric_text, re.S)
            if not param_match:
                raise ValueError("No logged configuration")
            logged = json.loads(param_match[-1])
            pid = (run["model"], run["dataset"], run["setting"], seed)
            if pid in params:
                raise ValueError("Duplicate formal run")
            params[pid] = {field: logged.get(field) for field in TRAIN_FIELDS}
            for field, expected in {"batch_size": batch, "seed": seed, "epochs": 1, "max_len": 100, "accumulation_steps": 1}.items():
                if int(logged[field]) != expected:
                    raise ValueError(f"Unexpected logged {field}: {pid}")
            if "_COMPLETE" not in slurm_text:
                raise ValueError("Missing completion marker")
            for label, values in parsed.items():
                ar = archived[run["model"], run["dataset"], run["setting"], label]
                if any(float(ar[name]) != val for name, val in values.items()):
                    raise ValueError("Archived metric differs from test log")
                mk = (run["model"], run["dataset"], label, run["setting"], seed)
                if mk in metrics:
                    raise ValueError("Duplicate raw metric")
                metrics[mk] = values
            per_rank = defaultdict(list)
            for rank, blocks, rows, batches in re.findall(r"\[Rank (\d+)\] blocked load balanced: blocks=(\d+), rows=(\d+), batches=(\d+)", slurm_text):
                per_rank[int(rank)].append((int(blocks), int(rows), int(batches)))
            if set(per_rank) != set(range(ws)) or any(len(v) % 3 or len(v) < 3 for v in per_rank.values()):
                raise ValueError(f"Expected train/valid/test allocations for every rank: {pid}")
            session_counts = {len(v) // 3 for v in per_rank.values()}
            if len(session_counts) != 1:
                raise ValueError(f"Unequal logged session counts across ranks: {pid}")
            logged_sessions = session_counts.pop()
            per_rank = {rank: values[-3:] for rank, values in per_rank.items()}
            for i, split in enumerate(("train", "valid", "test")):
                total = sum(values[i][1] for values in per_rank.values())
                processed = sum(values[i][2] for values in per_rank.values()) * batch
                coverage.append({"model": run["model"], "dataset": run["dataset"], "setting": run["setting"], "seed": seed,
                                 "split": split, "world_size": ws, "per_gpu_batch": batch, "assigned_rows": total,
                                 "processed_rows": processed, "dropped_rows": total-processed,
                                 "coverage_percent": processed/total*100,
                                 "rank_steps": "/".join(str(per_rank[r][i][2]) for r in range(ws)), "job_task_id": run["job_task_id"]})
            total_match = re.search(r"Total number of parameters: (\d+)", metric_text)
            dense_match = re.search(r"Number of dense parameters: (\d+)", metric_text)
            runs.append({"model": run["model"], "dataset": run["dataset"], "setting": run["setting"], "seed": seed,
                         "study": study, "job_task_id": run["job_task_id"], "gpu": run.get("gpu_name", run.get("gpu_type")),
                         "world_size": ws, "per_gpu_batch": batch, "global_batch": ws*batch,
                         "total_parameters": int(total_match[1]) if total_match else None,
                         "dense_parameters": int(dense_match[1]) if dense_match else None,
                         "metric_count": len(parsed), "original_metrics_match": True,
                         "logged_sessions": logged_sessions,
                         "state": run["state"], "exit_code": run["exit_code"], "slurm_log": run["slurm_log"], "checkpoint_log": run["checkpoint_log"]})

    assert len(runs) == 64 and len(metrics) == 272
    labels = []
    paper_path = repository / "experiments/sisa_three_seed_unified/sources/paper_table.csv"
    tracked(paper_path)
    paper = {key(r): r for r in read_csv(paper_path)}
    for model in MODELS:
        for dataset, names in LABELS.items():
            for label in names:
                baseline = metrics[model, dataset, label, "baseline", 20262027]
                row = {"model": model, "dataset": dataset, "label": label, "baseline_auc": baseline["AUC"], "baseline_logloss": baseline["logloss"],
                       "paper_auc": float(paper[model, dataset, label]["AUC"]), "paper_logloss": float(paper[model, dataset, label]["logloss"])}
                for seed in SEEDS:
                    value = metrics[model, dataset, label, "sisa", seed]
                    row.update({f"sisa_{seed}_auc": value["AUC"], f"sisa_{seed}_logloss": value["logloss"]})
                for metric in ("auc", "logloss"):
                    vals = [row[f"sisa_{seed}_{metric}"] for seed in SEEDS]
                    row[f"paired_delta_{metric}"] = vals[0] - row[f"baseline_{metric}"]
                    row[f"three_seed_mean_{metric}"] = statistics.fmean(vals)
                    row[f"three_seed_sd_{metric}"] = statistics.stdev(vals)
                    row[f"three_seed_delta_{metric}"] = statistics.fmean(vals) - row[f"baseline_{metric}"]
                labels.append(row)
    validate_keys(labels)
    cells, differences = [], []
    for model in MODELS:
        for dataset in LABELS:
            group = [r for r in labels if r["model"] == model and r["dataset"] == dataset]
            cell = {"model": model, "dataset": dataset, "dataset_display": NAMES[dataset], "labels": len(group)}
            for field in group[0]:
                if field not in ("model", "dataset", "label"):
                    cell[field] = statistics.fmean(r[field] for r in group)
            for metric in ("auc", "logloss"):
                vals = [cell[f"sisa_{seed}_{metric}"] for seed in SEEDS]
                cell[f"three_seed_sd_{metric}"] = statistics.stdev(vals)
                cell[f"three_seed_range_{metric}"] = max(vals)-min(vals)
            cell["paired_positive_auc_labels"] = sum(r["paired_delta_auc"] > 1e-12 for r in group)
            cell["three_seed_positive_auc_labels"] = sum(r["three_seed_delta_auc"] > 1e-12 for r in group)
            cells.append(cell)
            a, b = params[model, dataset, "baseline", 20262027], params[model, dataset, "sisa", 20262027]
            differences.extend({"model": model, "dataset": dataset, "field": f, "baseline": a[f], "sisa": b[f]} for f in TRAIN_FIELDS if a[f] != b[f])

    # Cross-check historical derived data without using it as the primary calculation.
    old_cells_path = repository / "experiments/sisa_three_seed_unified/results/cell_summary.csv"
    tracked(old_cells_path)
    old = {(r["model"], r["dataset"]): r for r in read_csv(old_cells_path)}
    for cell in cells:
        prior = old[cell["model"], cell["dataset"]]
        for new, previous in (("three_seed_mean_auc", "sisa_three_seed_mean_auc"), ("three_seed_sd_auc", "sisa_seed_std_auc"), ("three_seed_delta_logloss", "sisa_delta_logloss_vs_baseline")):
            if abs(cell[new] - float(prior[previous])) > 1e-12:
                raise ValueError("Historical summary disagrees with independent reconstruction")

    for model in MODELS:
        for dataset in LABELS:
            for split in ("train", "valid", "test"):
                selected = [r for r in coverage if r["model"] == model and r["dataset"] == dataset and r["split"] == split]
                old_pair = [r for r in selected if r["seed"] == 20262027]
                new_pair = [r for r in selected if r["seed"] != 20262027]
                for pair in (old_pair, new_pair):
                    if len({(r["assigned_rows"], r["processed_rows"], r["rank_steps"]) for r in pair}) != 1:
                        raise ValueError("Within-protocol coverage differs")

    segments = []
    for dimension, values in (("model", MODELS), ("dataset", LABELS)):
        for value in values:
            group = [r for r in cells if r[dimension] == value]
            segments.append({"dimension": dimension, "value": value,
                             **{f: statistics.fmean(r[f] for r in group) for f in ("paired_delta_auc", "paired_delta_logloss", "three_seed_delta_auc", "three_seed_delta_logloss")},
                             "paired_positive_auc_cells": sum(r["paired_delta_auc"] > 1e-12 for r in group)})
    resources = []
    for seed in (20262028, 20262029):
        audit_path = repository / f"experiments/sisa_single_seed{seed}/results/protocol_audit.md"
        audit_text = tracked(audit_path)
        for line in audit_text.splitlines():
            parts = [x.strip() for x in line.strip("|").split("|")]
            if len(parts) not in (5, 6) or not parts[0].isdigit() or parts[1] not in MODELS:
                continue
            h, m, s = map(int, parts[-2].split(":"))
            resources.append({"seed": seed, "task_id": int(parts[0]), "model": parts[1], "dataset": parts[2],
                              "elapsed_seconds": h*3600+m*60+s, "sampled_peak_memory_mib": int(parts[-1].replace(",", "")),
                              "gpu_hours": 2*(h*3600+m*60+s)/3600})
    assert len(resources) == 32
    counts = []
    for dataset in LABELS:
        row = {"dataset": dataset, "dataset_display": NAMES[dataset], "labels": len(LABELS[dataset])}
        for split in ("train", "valid", "test"):
            old = next(r for r in coverage if r["model"] == "HiFormer" and r["dataset"] == dataset and r["split"] == split and r["setting"] == "baseline")
            new = next(r for r in coverage if r["model"] == "HiFormer" and r["dataset"] == dataset and r["split"] == split and r["seed"] == 20262028)
            row.update({f"{split}_assigned": old["assigned_rows"], f"{split}_ws4_processed": old["processed_rows"], f"{split}_ws2_processed": new["processed_rows"]})
        counts.append(row)
    dataset_metadata = []
    for count in counts:
        dataset = count["dataset"]
        meta_path = repository / "datasets" / dataset / "meta_data.json"
        meta = json.loads(tracked(meta_path))
        saved = {"dataset": dataset, "metadata": {k: v for k, v in meta.items() if k in (
            "sample_size", "split_by_date", "split_strategy", "split_by_minimal_tail_with_positive_constraints",
            "user_filtering", "oov_filter", "sample_construction", "blocked_layout", "label")},
            "user_id_vocab_size": meta["vocab_size"]["user_id"], "item_id_vocab_size": meta["vocab_size"]["item_id"], "physical_splits": {}}
        for split in ("train", "valid", "test"):
            files = sorted((meta_path.parent / split / "data").glob("*.parquet"))
            file_rows = [{"path": p.relative_to(repository).as_posix(), "rows": pq.read_metadata(p).num_rows} for p in files]
            total = sum(r["rows"] for r in file_rows)
            if total != count[f"{split}_assigned"] or total != meta["sample_size"][split]:
                raise ValueError(f"Dataset metadata/footer/log row counts differ: {dataset}/{split}")
            saved["physical_splits"][split] = {"rows": total, "blocks": len(files), "files": file_rows}
        dataset_metadata.append(saved)

    summary = {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "accepted_models": list(MODELS), "accepted_datasets": list(LABELS),
               "formal_runs": len(runs), "verified_metric_pairs": len(metrics), "per_seed": [summarize(labels, seed) for seed in SEEDS],
               "three_seed_cell_macro_delta_auc": statistics.fmean(r["three_seed_delta_auc"] for r in cells),
               "three_seed_label_macro_delta_auc": statistics.fmean(r["three_seed_delta_auc"] for r in labels),
               "three_seed_cell_macro_delta_logloss": statistics.fmean(r["three_seed_delta_logloss"] for r in cells),
               "same_protocol_mean_abs_cell_seed_delta_auc": statistics.fmean(abs(r["sisa_20262029_auc"]-r["sisa_20262028_auc"]) for r in cells),
               "baseline_paper_auc_mae": statistics.fmean(abs(r["baseline_auc"]-r["paper_auc"]) for r in labels),
               "baseline_paper_auc_max_abs": max(abs(r["baseline_auc"]-r["paper_auc"]) for r in labels),
               "source_metric_validation": "All 272 archived AUC/logloss pairs match explicit final-test log values; all 64 runs have completion and valid protocol evidence.",
               "historical_summary_validation": "Independent reconstructed three-seed means, sample SDs and logloss deltas agree within 1e-12.",
               "coverage_validation": "All 16 old baseline/SISA pairs and all 16 new SISA seed pairs match per-rank train/valid/test allocation counts and steps.",
               "dataset_rows_validation": "All 12 split row counts agree among current Parquet footers, dataset metadata and archived loader allocations; no full-content hash or event-level audit performed.",
               "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()}
    current_source = repository / "unirank/pytorch/layers/attentions/sisa.py"
    try:
        ast.parse(current_source.read_text())
        summary["current_sisa_syntax"] = "passed"
    except SyntaxError as error:
        summary["current_sisa_syntax"] = f"pre-existing SyntaxError at line {error.lineno}: {error.msg}"
    for name, rows in (("label_results", labels), ("cell_results", cells), ("run_evidence", runs), ("coverage_audit", coverage),
                       ("dataset_counts", counts), ("segment_results", segments), ("paired_config_differences", differences), ("resource_usage", resources)):
        write_csv(result_dir / f"{name}.csv", rows)
    for name, obj in (("summary", summary), ("dataset_metadata", dataset_metadata), ("logged_configurations", [{"model": k[0], "dataset": k[1], "setting": k[2], "seed": k[3], "params": v} for k,v in params.items()])):
        (result_dir / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n")
    for rel in ("AGENTS.md", "config/model_config.yaml", "config/dataset_config.yaml", "uv.lock", "pyproject.toml", "run_expid.py", "unirank/pytorch/layers/attentions/sisa.py", "unirank/pytorch/models/multitask_model.py", "unirank/pytorch/dataloaders/unirank_dataloader.py"):
        tracked(repository / rel)
    write_csv(result_dir / "source_manifest.csv", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repository_root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "pyproject.toml").is_file() and (parent / "unirank").is_dir()
    )
    parser.add_argument("--repository", type=Path, default=repository_root)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/sisa_paper_report/archive"))
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else repository / args.output_dir
    build(repository, output)


if __name__ == "__main__":
    main()
