"""Reproduce the HiFormer/QK diagnosis from existing local evidence only."""

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parents[1] / "results"
PAPER = ROOT / "experiments/sisa_paper_report/archive/results"
LABELS = ("click", "follow", "like", "share")
METRIC = re.compile(r"\[Task: (\w+)\]\[Metrics\] (.*)")
VALUE = re.compile(r"(AUC|logloss): ([\d.]+)")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_log(path):
    text = path.read_text()
    config, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
    assert text.count("Test evaluation") == 1, path
    assert text.count("Evaluation @epoch") == 1, path
    parts = text.split("Test evaluation")
    metrics = {}
    for split, part in zip(("valid", "test"), parts):
        metrics[split] = {
            label: {key: float(value) for key, value in VALUE.findall(values)}
            for label, values in METRIC.findall(part)
        }
        assert set(metrics[split]) == set(LABELS), path
    return config, metrics


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [
        row for row in read_csv(PAPER / "run_evidence.csv")
        if row["model"] == "HiFormer" and row["dataset"] == "QK_Video_Action"
    ]
    assert len(rows) == 4
    labels = {
        row["label"]: row for row in read_csv(PAPER / "label_results.csv")
        if row["model"] == "HiFormer" and row["dataset"] == "QK_Video_Action"
    }
    manifest = {row["path"]: row for row in read_csv(PAPER / "source_manifest.csv")}
    coverage = [
        row for row in read_csv(PAPER / "coverage_audit.csv")
        if row["model"] == "HiFormer" and row["dataset"] == "QK_Video_Action"
    ]
    observations, runs = [], []
    parsed = {}
    for row in rows:
        path = ROOT / row["checkpoint_log"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == manifest[row["checkpoint_log"]]["sha256"], path
        config, metrics = parse_log(path)
        key = row["setting"], row["seed"]
        parsed[key] = config, metrics
        prefix = "baseline" if row["setting"] == "baseline" else f"sisa_{row['seed']}"
        for label in LABELS:
            for metric, column in (("AUC", "auc"), ("logloss", "logloss")):
                assert math.isclose(
                    metrics["test"][label][metric],
                    float(labels[label][f"{prefix}_{column}"]),
                    rel_tol=0, abs_tol=1e-12,
                ), (path, label, metric)
        for split in ("valid", "test"):
            for label in LABELS:
                observations.append({
                    "setting": row["setting"], "seed": row["seed"],
                    "split": split, "label": label,
                    **metrics[split][label], "source": row["checkpoint_log"],
                })
        runs.append({
            "setting": row["setting"], "seed": row["seed"],
            "source": row["checkpoint_log"], "sha256": digest,
            "job_task_id": row["job_task_id"], "config": config,
            "macro": {
                split: {
                    metric: sum(metrics[split][label][metric] for label in LABELS) / 4
                    for metric in ("AUC", "logloss")
                } for split in ("valid", "test")
            },
        })
    baseline, baseline_metrics = parsed[("baseline", "20262027")]
    sisa, sisa_metrics = parsed[("sisa", "20262027")]
    config_diff = {
        key: [baseline.get(key), sisa.get(key)]
        for key in sorted(set(baseline) | set(sisa))
        if baseline.get(key) != sisa.get(key)
    }
    assert set(config_diff) == {
        "model_id", "sisa_enabled", "sisa_lambda_init",
        "sisa_score_dim", "sisa_score_scale",
    }
    for split in ("train", "valid", "test"):
        pair = [row for row in coverage if row["seed"] == "20262027" and row["split"] == split]
        assert len(pair) == 2
        for key in ("world_size", "per_gpu_batch", "assigned_rows", "processed_rows", "rank_steps"):
            assert pair[0][key] == pair[1][key], (split, key)
    delta_rows = []
    for label in LABELS:
        before, after = baseline_metrics["test"][label], sisa_metrics["test"][label]
        delta_rows.append({
            "label": label, "baseline_auc": before["AUC"], "sisa_auc": after["AUC"],
            "delta_auc": after["AUC"] - before["AUC"],
            "macro_auc_contribution": (after["AUC"] - before["AUC"]) / 4,
            "baseline_logloss": before["logloss"], "sisa_logloss": after["logloss"],
            "delta_logloss": after["logloss"] - before["logloss"],
        })
    net_delta = sum(row["macro_auc_contribution"] for row in delta_rows)
    follow_share = next(row["macro_auc_contribution"] for row in delta_rows if row["label"] == "follow") / net_delta

    import torch

    scales = []
    for path in sorted((ROOT / "checkpoints/QK_Video_Action").glob("*HiFormer*.model")):
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        for name, tensor in state.items():
            if name.endswith("raw_head_scale"):
                for head, value in enumerate((torch.nn.functional.softplus(tensor.float()) + 1e-8).tolist()):
                    scales.append({
                        "checkpoint": str(path.relative_to(ROOT)),
                        "tensor": name, "head": head, "lambda": value,
                    })
        del state
    code_sources = ["model_zoo/HiFormer.py", "unirank/pytorch/layers/attentions/sisa.py", "run_expid.py"]
    evidence = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "HiFormer / QK_Video_Action",
        "runs": runs, "paired_config_differences": config_diff,
        "coverage": coverage, "paired_test_delta_auc": net_delta,
        "follow_share_of_net_auc_decline": follow_share,
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in code_sources},
        "checks": {
            "all_four_log_hashes_match_paper_archive": True,
            "all_32_test_metrics_match_paper_archive": True,
            "paired_logged_backbone_config_matches": True,
            "paired_archived_coverage_matches": True,
            "one_validation_candidate_per_run": True,
        },
        "limitations": [
            "20262028/29 differ in world size, batch size and coverage; not matched baseline replications.",
            "Coverage is cross-checked against the existing audit, not recomputed from all dataset rows.",
            "Learned scales come from 20262028/29 checkpoints, not the deleted matched-run checkpoint.",
            "Head lambda alone does not measure effective score-bias magnitude or causal utility.",
        ],
    }
    (OUTPUT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    write_csv(OUTPUT / "label_metrics.csv", observations)
    write_csv(OUTPUT / "paired_label_deltas.csv", delta_rows)
    if scales:
        write_csv(OUTPUT / "learned_head_scales.csv", scales)
    print(json.dumps({
        "runs": len(runs), "matched_test_delta_auc": net_delta,
        "follow_share_of_net_decline": follow_share, "checks": evidence["checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
