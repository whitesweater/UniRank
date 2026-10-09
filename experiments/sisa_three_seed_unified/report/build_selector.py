"""Build the self-contained SISA run selector from audited experiment evidence."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MODELS = ("HiFormer", "HyFormer", "OneTrans", "RankMixer", "Zenith")
DATASETS = (
    ("QK_Video_Action", "QK-Video"),
    ("KuaiRand_Video_Action", "KuaiRand"),
    ("TencentGR_10M_Action", "TAAC-25"),
    ("MerRec_Action", "MerRec"),
)
LABEL_COUNT = dict(zip((dataset for dataset, _ in DATASETS), (4, 6, 2, 5)))
SOURCES = (
    "experiments/sisa_three_seed_unified/results/label_comparison.csv",
    "experiments/sisa_three_seed_unified/results/integrated_cells.csv",
    "experiments/sisa_seed_variability_20260923/results/verified_label_metrics.csv",
    "experiments/sisa_seed_variability_20260923/results/run_log_audit.csv",
    "experiments/sisa_seed_variability_20260923/results/final_comparison.csv",
    "experiments/sisa_seed_variability_20260923/rerun/results/label_comparison.csv",
    "experiments/sisa_seed_variability_20260923/rerun/followup_20260923/results/label_comparison.csv",
    "experiments/onetrans_seed_completion/results/selected_label_metrics.csv",
    "experiments/onetrans_seed_completion/results/selected_runs.csv",
    "experiments/sisa_native_strict/results/runs.csv",
    "experiments/sisa_expansion_acd/results/runs.csv",
    "experiments/hiformer_target_only_multiseed/results/runs.csv",
    "experiments/hiformer_target_only_multiseed/results/label_metrics.csv",
    "experiments/hiformer_target_only_multiseed/results/summary.csv",
    "experiments/hiformer_target_only_multiseed/notes/independent_final_audit.json",
)


def read_csv(root: Path, relative: str) -> list[dict[str, str]]:
    with (root / relative).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def add_label(groups: dict, key: tuple, label: str, auc: str, logloss: str) -> None:
    values = {"label": label, "auc": float(auc), "logloss": float(logloss)}
    assert math.isfinite(values["auc"]) and 0 <= values["auc"] <= 1
    assert math.isfinite(values["logloss"]) and values["logloss"] >= 0
    assert label not in groups[key], (key, label)
    groups[key][label] = values


def ordered_labels(group: dict[str, dict]) -> list[dict]:
    return [group[label] for label in sorted(group)]


def make_run(model: str, dataset: str, seed: int, job: str, kind: str,
             labels: list[dict], source: str, protocol: str = "") -> dict:
    assert len(labels) == LABEL_COUNT[dataset]
    return {
        "id": f"{model}|{dataset}|{seed}|{job}",
        "seed": seed,
        "job": job,
        "kind": kind,
        "auc": statistics.fmean(row["auc"] for row in labels),
        "logloss": statistics.fmean(row["logloss"] for row in labels),
        "labels": labels,
        "source": source,
        "protocol": protocol,
    }


def collect_catalog(root: Path = ROOT) -> dict:
    root = root.resolve()
    accepted = {(model, dataset) for model in MODELS for dataset, _ in DATASETS}
    baselines: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in read_csv(root, SOURCES[0]):
        key = row["model"], row["dataset"]
        assert key in accepted and key[0] != "OneTrans"
        add_label(baselines, key, row["label"], row["baseline_auc"],
                  row["baseline_logloss"])
    for row in read_csv(root, SOURCES[7]):
        if row["setting"] == "baseline":
            key = "OneTrans", row["dataset"]
            assert key in accepted
            add_label(baselines, key, row["label"], row["AUC"], row["logloss"])
    assert len(baselines) == 20

    baseline_jobs = {}
    for relative in (SOURCES[9], SOURCES[10]):
        for row in read_csv(root, relative):
            key = row.get("model"), row.get("dataset")
            if key not in accepted or key[0] == "OneTrans" or row.get("setting") != "baseline":
                continue
            assert row.get("complete") == "True"
            assert key not in baseline_jobs
            baseline_jobs[key] = (row["job_task_id"], relative)
    for row in read_csv(root, SOURCES[8]):
        if row["setting"] == "baseline":
            key = "OneTrans", row["dataset"]
            assert key in accepted and row["eligible"] == "True"
            assert key not in baseline_jobs
            baseline_jobs[key] = (row["job_task_id"], SOURCES[8])
    assert len(baseline_jobs) == 20

    original_labels: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in read_csv(root, SOURCES[2]):
        key = row["model"], row["dataset"], int(row["seed"])
        assert key[:2] in accepted
        add_label(original_labels, key, row["label"], row["AUC"], row["logloss"])
    audits = {}
    for row in read_csv(root, SOURCES[3]):
        key = row["model"], row["dataset"], int(row["seed"])
        assert row["final_test_matches_selected_metrics"] == "True"
        assert key not in audits
        audits[key] = row
    assert len(original_labels) == len(audits) == 60

    target_audit = json.loads((root / SOURCES[14]).read_text())
    assert target_audit["passed"] is True and target_audit["target_results_checked"] == 12
    target_runs = {}
    for row in read_csv(root, SOURCES[11]):
        assert row["model"] == "HiFormer" and row["mode"] == "target_only"
        assert row["eligible"] == "True" and row["state"] == "COMPLETED" and row["exit_code"] == "0:0"
        key = row["dataset"], int(row["seed"])
        assert key not in target_runs
        target_runs[key] = row
    target_labels: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in read_csv(root, SOURCES[12]):
        key = row["dataset"], int(row["seed"])
        assert key in target_runs and target_runs[key]["job_task_id"] == row["job_task_id"]
        add_label(target_labels, key, row["label"], row["AUC"], row["logloss"])
    target_summaries = {row["dataset"]: row for row in read_csv(root, SOURCES[13])}
    assert len(target_runs) == len(target_labels) == 12 and len(target_summaries) == 4

    extra_labels: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for relative in (SOURCES[5], SOURCES[6]):
        for row in read_csv(root, relative):
            key = row["model"], row["dataset"], int(row["seed"]), row["job_task_id"]
            values = {"label": row["label"], "auc": float(row["new_auc"]),
                      "logloss": float(row["new_logloss"])}
            prior = extra_labels[key].get(row["label"])
            if prior is not None:
                assert prior == values, (key, row["label"])
            else:
                extra_labels[key][row["label"]] = values
    attempts = {}
    for row in read_csv(root, SOURCES[4]):
        key = row["model"], row["dataset"], int(row["run_seed"]), row["job_task_id"]
        assert key in extra_labels and key not in attempts
        attempts[key] = row
    assert len(extra_labels) == len(attempts) == 10

    summaries = {}
    for row in read_csv(root, SOURCES[1]):
        key = row["model"], row["dataset_id"]
        assert key in accepted and key not in summaries
        summaries[key] = row
    assert len(summaries) == 20

    cells = []
    for model in MODELS:
        for dataset, name in DATASETS:
            key = model, dataset
            job, source = baseline_jobs[key]
            baseline = make_run(model, dataset, 20262027, job, "baseline",
                                ordered_labels(baselines[key]), source)
            recorded = summaries[key]
            assert abs(baseline["auc"] - float(recorded["baseline_auc"])) < 1e-12
            assert abs(baseline["logloss"] - float(recorded["baseline_logloss"])) < 1e-12
            runs = []
            for seed in (20262027, 20262028, 20262029):
                audit = audits[model, dataset, seed]
                run = make_run(model, dataset, seed, audit["job_task_id"], "original",
                               ordered_labels(original_labels[model, dataset, seed]),
                               audit["checkpoint_log"])
                assert abs(run["auc"] - float(recorded[f"original_{seed}_auc"])) < 1e-12
                assert abs(run["logloss"] - float(recorded[f"original_{seed}_logloss"])) < 1e-12
                runs.append(run)
            for (m, d, seed, replay_job), labels in extra_labels.items():
                if (m, d) != key:
                    continue
                attempt = attempts[m, d, seed, replay_job]
                kind = "new_seed" if seed == 20262030 else "rerun"
                run = make_run(m, d, seed, replay_job, kind,
                               ordered_labels(labels), attempt["checkpoint_log"])
                assert abs(run["auc"] - float(attempt["run_auc"])) < 1e-12
                assert abs(run["logloss"] - float(attempt["run_logloss"])) < 1e-12
                runs.append(run)
            target_mean = None
            if model == "HiFormer":
                target_values = []
                for seed in (20262027, 20262028, 20262029):
                    target = target_runs[dataset, seed]
                    run = make_run(model, dataset, seed, target["job_task_id"], "target_only",
                                   ordered_labels(target_labels[dataset, seed]),
                                   target["checkpoint_log"],
                                   f"ws{target['world_size']}_bs{target['batch_size']}")
                    assert abs(run["auc"] - float(target["macro_auc"])) < 1e-12
                    assert abs(run["logloss"] - float(target["macro_logloss"])) < 1e-12
                    runs.append(run)
                    target_values.append(run["auc"])
                target_mean = statistics.fmean(target_values)
                assert abs(target_mean - float(target_summaries[dataset]["target_only_auc_mean"])) < 1e-12
            assert len({run["id"] for run in runs}) == len(runs)
            labels_expected = {label["label"] for label in baseline["labels"]}
            assert all({label["label"] for label in run["labels"]} == labels_expected
                       for run in runs)
            cells.append({
                "id": f"{model}|{dataset}",
                "model": model,
                "dataset": dataset,
                "dataset_name": name,
                "baseline": baseline,
                "runs": sorted(runs, key=lambda r: (r["seed"],
                                                     0 if r["kind"] == "original" else 1)),
                "original_three_seed_auc": float(recorded["original_three_seed_mean_auc"]),
                "original_three_seed_logloss": float(recorded["original_three_seed_mean_logloss"]),
                "target_only_three_seed_auc": target_mean,
            })
    assert len(cells) == 20
    assert sum(len(cell["runs"]) for cell in cells) == 82
    assert sum(len(cell["baseline"]["labels"]) for cell in cells) == 85
    sha = hashlib.sha256()
    sources = []
    for relative in SOURCES:
        content = (root / relative).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        sha.update(relative.encode() + bytes.fromhex(digest))
        sources.append({"file": relative, "sha256": digest})
    for cell in cells:
        cell["default_selection"] = default_selection(cell)
        if cell["model"] == "HiFormer":
            cell["full_default_selection"] = best_selection(cell, "full")
    return {
        "version": 1,
        "selection_policy": "auc_top3_distinct_v1",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "digest": sha.hexdigest(),
        "models": list(MODELS),
        "datasets": [{"id": dataset, "name": name} for dataset, name in DATASETS],
        "cells": cells,
        "sources": sources,
    }


def best_selection(cell: dict, method: str = "target_only") -> dict[str, str]:
    options = [run for run in cell["runs"] if cell["model"] != "HiFormer"
               or (run["kind"] == "target_only") == (method == "target_only")]
    best_by_seed = {}
    for run in options:
        prior = best_by_seed.get(run["seed"])
        if prior is None or run["auc"] > prior["auc"]:
            best_by_seed[run["seed"]] = run
    winners = sorted(best_by_seed.values(), key=lambda run: (-run["auc"], run["seed"]))[:3]
    assert len(winners) == 3 and len({run["seed"] for run in winners}) == 3
    return {str(run["seed"]): run["id"] for run in sorted(winners, key=lambda run: run["seed"])}


def default_selection(cell: dict) -> dict[str, str]:
    return best_selection(cell)


def main() -> None:
    catalog = collect_catalog(ROOT)
    old_catalog_path = HERE / "selector_catalog.json"
    if old_catalog_path.exists():
        previous = json.loads(old_catalog_path.read_text(encoding="utf-8"))
        if previous.get("digest") != catalog["digest"]:
            catalog["previous_digest"] = previous["digest"]
        elif previous.get("previous_digest"):
            catalog["previous_digest"] = previous["previous_digest"]
    payload = json.dumps(catalog, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = (HERE / "selector_template.html").read_text(encoding="utf-8")
    assert template.count("__CATALOG_JSON__") == 1
    assert template.count("__FALLBACK_TABLE__") == 1
    cells = [
        "<table class=\"matrix\"><thead><tr><th>模型</th>"
        + "".join(f"<th>{html.escape(name)}</th>" for _, name in DATASETS)
        + "</tr></thead><tbody>"
    ]
    indexed = {(cell["model"], cell["dataset"]): cell for cell in catalog["cells"]}
    for model in MODELS:
        cells.append(f"<tr><th>{html.escape(model)}</th>")
        for dataset, _ in DATASETS:
            cell = indexed[model, dataset]
            cells.append(
                f"<td>Baseline {cell['baseline']['auc']:.7f}<br>"
                f"SISA {statistics.fmean(next(run['auc'] for run in cell['runs'] if run['id'] == run_id) for run_id in default_selection(cell).values()):.7f}</td>"
            )
        cells.append("</tr>")
    cells.append("</tbody></table>")
    report = (template.replace("__CATALOG_JSON__", payload)
              .replace("__FALLBACK_TABLE__", "".join(cells)))
    (HERE / "report.html").write_text(report, encoding="utf-8")
    (HERE / "selector_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Built selector: {len(catalog['cells'])} cells, "
          f"{sum(len(c['runs']) for c in catalog['cells'])} SISA run choices, "
          "20 Baselines; report.html is self-contained.")


if __name__ == "__main__":
    main()
