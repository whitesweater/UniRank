"""Audit existing RISE evidence without training or model inference.

Run with the repository environment from any working directory. Outputs are
separate from the historical reports and never replace their frozen evidence.
"""

import csv
import hashlib
import json
import re
import statistics
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parents[1] / "results"
PAPER = ROOT / "experiments/sisa_paper_report/archive/results"
COMPONENT = ROOT / "experiments/ablations/sisa_component_report/archive/results"
DATASETS = (
    "QK_Video_Action", "KuaiRand_Video_Action",
    "TencentGR_10M_Action", "MerRec_Action",
)
MODELS = ("HiFormer", "HyFormer", "RankMixer", "Zenith")
SOURCES = set()


def rows(path):
    SOURCES.add(path)
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, values):
    with (OUTPUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def parse_log(path):
    SOURCES.add(path)
    raw = path.read_text()
    assert raw.count("Test evaluation") == 1, path
    params, _ = json.JSONDecoder().raw_decode(raw[raw.index("{"):])
    tail = raw.rsplit("Test evaluation", 1)[1]
    metrics = {
        label: {key: float(value) for key, value in
                re.findall(r"(AUC|logloss): ([\d.]+)", body)}
        for label, body in re.findall(r"\[Task: (\w+)\]\[Metrics\] (.*)", tail)
    }
    return params, metrics, "Deleted model checkpoint" in tail


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archived_runs = rows(PAPER / "run_evidence.csv")
    archived_labels = {
        (r["model"], r["dataset"], r["label"]): r
        for r in rows(PAPER / "label_results.csv")
    }
    parsed, inventory, verified = {}, [], 0
    for run in archived_runs:
        checkpoint = (ROOT / run["checkpoint_log"]).with_suffix(".model")
        deleted = "not_reparsed"
        if run["seed"] == "20262027":
            _, metrics, deleted = parse_log(ROOT / run["checkpoint_log"])
            prefix = "baseline" if run["setting"] == "baseline" else "sisa_20262027"
            for label, values in metrics.items():
                reference = archived_labels[run["model"], run["dataset"], label]
                for metric, field in (("AUC", "auc"), ("logloss", "logloss")):
                    assert abs(values[metric] - float(reference[f"{prefix}_{field}"])) < 1e-12
                    verified += 1
            parsed[run["model"], run["dataset"], run["setting"]] = metrics
        inventory.append({
            "model": run["model"], "dataset": run["dataset"],
            "setting": run["setting"], "seed": run["seed"],
            "job_task_id": run["job_task_id"],
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "exists": checkpoint.is_file(),
            "bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
            "deletion_in_reparsed_log": deleted,
        })
    assert len(parsed) == 32 and verified == 272
    paired_labels, paired_cells, datasets = [], [], []
    for dataset in DATASETS:
        for model in MODELS:
            baseline = parsed[model, dataset, "baseline"]
            rise = parsed[model, dataset, "sisa"]
            assert baseline.keys() == rise.keys()
            cell = {"model": model, "dataset": dataset, "labels": len(baseline)}
            for metric in ("AUC", "logloss"):
                cell[f"baseline_{metric}"] = statistics.mean(x[metric] for x in baseline.values())
                cell[f"rise_{metric}"] = statistics.mean(x[metric] for x in rise.values())
                cell[f"delta_{metric}"] = cell[f"rise_{metric}"] - cell[f"baseline_{metric}"]
            paired_cells.append(cell)
            for label in baseline:
                row = {"model": model, "dataset": dataset, "label": label}
                for metric in ("AUC", "logloss"):
                    row[f"baseline_{metric}"] = baseline[label][metric]
                    row[f"rise_{metric}"] = rise[label][metric]
                    row[f"delta_{metric}"] = rise[label][metric] - baseline[label][metric]
                paired_labels.append(row)
        cells = [r for r in paired_cells if r["dataset"] == dataset]
        datasets.append({"dataset": dataset, **{
            key: statistics.mean(r[key] for r in cells)
            for key in ("baseline_AUC", "rise_AUC", "delta_AUC", "delta_logloss")
        }})

    component_runs = [r for r in rows(COMPONENT / "runs.csv")
                      if r["model"] == "HyFormer" and r["dataset"] == "MerRec_Action"]
    reference_metrics = {(r["setting"], r["label"]): r
                         for r in rows(COMPONENT / "label_metrics.csv")
                         if r["model"] == "HyFormer" and r["dataset"] == "MerRec_Action"}
    component_params, component_metrics, component_inventory = {}, {}, []
    for run in component_runs:
        setting = run["setting"]
        assert run["eligible"] == "True" and run["state"] == "COMPLETED"
        params, metrics, _ = parse_log(ROOT / run["checkpoint_log"])
        component_params[setting], component_metrics[setting] = params, metrics
        for label, values in metrics.items():
            for metric in ("AUC", "logloss"):
                assert abs(values[metric] - float(reference_metrics[setting, label][metric])) < 1e-12
        path = (ROOT / run["checkpoint_log"]).with_suffix(".model")
        assert path.is_file()
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata = archive.read(next(name for name in names if name.endswith("data.pkl")))
        component_inventory.append({
            "setting": setting, "job_task_id": run["job_task_id"],
            "checkpoint": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "zip_entries": len(names), "compiled_key_metadata": b"_orig_mod" in metadata,
            "dense_parameters": run["dense_parameters"],
            "macro_AUC": statistics.mean(x["AUC"] for x in metrics.values()),
            "macro_logloss": statistics.mean(x["logloss"] for x in metrics.values()),
        })
    assert set(component_params) == {"B", "C", "D", "E"}
    differing = [key for key in sorted(set().union(*(p.keys() for p in component_params.values())))
                 if len({str(p.get(key)) for p in component_params.values()}) > 1]
    assert set(differing) == {"model_id", "sisa_variant"}, differing
    component_effects = []
    for label in component_metrics["B"]:
        values = {s: component_metrics[s][label]["AUC"] for s in "BCDE"}
        component_effects.append({
            "label": label, **{f"{s}_AUC": values[s] for s in "BCDE"},
            "C_minus_B": values["C"] - values["B"],
            "D_minus_B": values["D"] - values["B"],
            "E_minus_B": values["E"] - values["B"],
            "E_minus_C": values["E"] - values["C"],
            "E_minus_D": values["E"] - values["D"],
            "interaction_E_C_D_B": values["E"] - values["C"] - values["D"] + values["B"],
        })

    rng = np.random.default_rng(20260923)
    history, sampled_parts = [], []
    for dataset in DATASETS:
        paths = sorted((ROOT / "datasets" / dataset / "test/data").glob("*.parquet"))
        assert paths
        total, extrema, sampled, saturated_estimate = 0, [], 0, 0.0
        for path in paths:
            parquet = pq.ParquetFile(path)
            index = parquet.schema_arrow.get_field_index("seq_len")
            assert index >= 0
            meta = parquet.metadata
            counts = np.array([meta.row_group(i).num_rows for i in range(meta.num_row_groups)])
            for i in range(meta.num_row_groups):
                stats = meta.row_group(i).column(index).statistics
                assert stats and stats.has_min_max
                extrema.extend([int(stats.min), int(stats.max)])
            group = int(rng.choice(meta.num_row_groups, p=counts / counts.sum()))
            lengths = parquet.read_row_group(group, columns=["seq_len"]).column(0).to_numpy()
            chosen = rng.choice(len(lengths), size=min(4096, len(lengths)), replace=False)
            subset = lengths[chosen]
            ratio = float(np.mean(subset >= 100))
            total += meta.num_rows
            sampled += len(subset)
            saturated_estimate += meta.num_rows * ratio
            sampled_parts.append({
                "dataset": dataset, "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                "rows": meta.num_rows, "row_group": group,
                "sample_rows": len(subset), "sample_ge100": int(np.sum(subset >= 100)),
                "sample_offsets_sha256": hashlib.sha256(chosen.astype("<i8").tobytes()).hexdigest(),
            })
        history.append({"dataset": dataset, "test_rows_footer": total,
                        "seq_len_min_footer": min(extrema), "seq_len_max_footer": max(extrema),
                        "sample_rows": sampled, "estimated_share_ge100": saturated_estimate / total})

    for name, data in (("paired_labels.csv", paired_labels), ("paired_cells.csv", paired_cells),
                       ("paired_datasets.csv", datasets), ("checkpoint_inventory.csv", inventory),
                       ("component_checkpoints.csv", component_inventory),
                       ("component_effects.csv", component_effects),
                       ("history_feasibility.csv", history), ("history_sample_manifest.csv", sampled_parts)):
        write_csv(name, data)
    SOURCES.add(ROOT / "document.pdf")
    SOURCES.add(Path(__file__).resolve())
    source_manifest = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in sorted(SOURCES)]
    write_csv("source_manifest.csv", source_manifest)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "paired_logs_reparsed": len(parsed), "paired_metric_values_verified": verified,
        "component_logs_reparsed": len(component_metrics), "component_metric_values_verified": 40,
        "component_differing_config_keys": differing,
        "missing_main_pair_checkpoints": sum(r["seed"] == "20262027" and not r["exists"] for r in inventory),
        "explicit_main_checkpoint_deletions": sum(r["deletion_in_reparsed_log"] is True for r in inventory),
        "cell_macro_delta_auc": statistics.mean(r["delta_AUC"] for r in paired_cells),
        "positive_auc_cells": sum(r["delta_AUC"] > 0 for r in paired_cells),
        "positive_auc_labels": sum(r["delta_AUC"] > 0 for r in paired_labels),
        "component_macro_effects": {key: statistics.mean(r[key] for r in component_effects)
                                   for key in component_effects[0] if key != "label"},
        "history_sampling_seed": 20260923,
        "history_sampling_design": "One row group per test part, selected proportional to row count; up to 4096 rows without replacement; part estimates weighted by full part row counts.",
        "history_sampling_limit": "Feasibility diagnostic only; one sampled row group per part can miss within-part heterogeneity. Not a full distribution or inferential result.",
        "model_loading_or_forward_executed": False,
        "training_executed": False,
        "historical_data_full_hash_verified": False,
    }
    (OUTPUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
