"""Recompute seed spreads from final Test logs; run with repository Python."""

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path.cwd()
while not (ROOT / "run_expid.py").exists():
    if ROOT == ROOT.parent:
        raise RuntimeError("Run inside the UniRank repository")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from scripts.collect_sisa_single_seed_results import parse_test_metrics

OUT = ROOT / "experiments/sisa_seed_variability_20260923"
(OUT / "results").mkdir(parents=True, exist_ok=True)
SEEDS = (20262027, 20262028, 20262029)
MODELS = ("HiFormer", "HyFormer", "RankMixer", "Zenith", "OneTrans")
DATASETS = {
    "QK_Video_Action": "QK-Video",
    "KuaiRand_Video_Action": "KuaiRand",
    "TencentGR_10M_Action": "TAAC-25",
    "MerRec_Action": "MerRec",
}
COUNTS = dict(zip(DATASETS, (4, 6, 2, 5)))
sources = []


def read_csv(relative):
    path = ROOT / relative
    sources.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUT / "results" / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


wide = read_csv("experiments/sisa_three_seed_unified/results/label_comparison.csv")
expected = {}
for row in wide:
    for seed in SEEDS:
        expected[row["model"], row["dataset"], seed, row["label"]] = {
            "AUC": float(row[f"sisa_{seed}_auc"]),
            "logloss": float(row[f"sisa_{seed}_logloss"]),
        }
for row in read_csv("experiments/onetrans_seed_completion/results/selected_label_metrics.csv"):
    if row["setting"] == "sisa":
        expected["OneTrans", row["dataset"], int(row["seed"]), row["label"]] = {
            metric: float(row[metric]) for metric in ("AUC", "logloss")
        }
assert len(expected) == 255

runs = []
for study in ("sisa_native_strict", "sisa_expansion_acd"):
    for row in read_csv(f"experiments/{study}/results/runs.csv"):
        m, d = row["model"], row["dataset"]
        if row["setting"] != "sisa" or m not in MODELS[:-1] or d not in DATASETS:
            continue
        use_strict = m != "HyFormer" and d != "TencentGR_10M_Action"
        if use_strict == (study == "sisa_native_strict"):
            runs.append(dict(row, seed=20262027))
for seed in SEEDS[1:]:
    runs.extend(read_csv(f"experiments/sisa_single_seed{seed}/results/runs.csv"))
for row in read_csv("experiments/onetrans_seed_completion/results/selected_runs.csv"):
    if row["setting"] == "sisa":
        runs.append(row)
assert len(runs) == 60
assert len({(r["model"], r["dataset"], int(r["seed"])) for r in runs}) == 60

audit, long_rows = [], []
for row in runs:
    m, d, seed = row["model"], row["dataset"], int(row["seed"])
    assert row.get("eligible", row.get("complete")) == "True", row
    parsed = []
    receipt = {"model": m, "dataset": d, "seed": seed, "job_task_id": row["job_task_id"],
               "archived_state": row["state"]}
    for field in ("checkpoint_log", "slurm_log"):
        path = ROOT / row[field]
        text = path.read_text(errors="replace")
        metrics = parse_test_metrics(text)
        assert len(metrics) == COUNTS[d], (m, d, seed, field, metrics)
        assert set(metrics) == {key[3] for key in expected if key[:3] == (m, d, seed)}
        for label, values in metrics.items():
            for metric in ("AUC", "logloss"):
                value = values[metric]
                assert math.isfinite(value)
                assert abs(value - expected[m, d, seed, label][metric]) < 1e-12
        parsed.append(metrics)
        receipt[field] = row[field]
        receipt[field + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert parsed[0] == parsed[1], (m, d, seed)
    receipt["final_test_matches_selected_metrics"] = True
    audit.append(receipt)
    for label, values in parsed[0].items():
        long_rows.append({"model": m, "dataset": d, "label": label, "seed": seed,
                          "AUC": values["AUC"], "logloss": values["logloss"]})


def describe(base, metric, values):
    for seed, value in zip(SEEDS, values):
        base[f"{metric}_{seed}"] = value
    base[f"{metric}_mean"] = statistics.fmean(values)
    base[f"{metric}_sample_sd"] = statistics.stdev(values)
    base[f"{metric}_range"] = max(values) - min(values)
    best = max(values) if metric == "auc" else min(values)
    worst = min(values) if metric == "auc" else max(values)
    base[f"{metric}_best_seed"] = SEEDS[values.index(best)]
    base[f"{metric}_worst_seed"] = SEEDS[values.index(worst)]
    if metric == "logloss":
        base["logloss_relative_range"] = (max(values) - min(values)) / min(values)


labels = []
groups = defaultdict(list)
for row in long_rows:
    groups[row["model"], row["dataset"], row["label"]].append(row)
for (m, d, label), rows in groups.items():
    rows.sort(key=lambda row: row["seed"])
    assert tuple(row["seed"] for row in rows) == SEEDS
    base = {"model": m, "dataset": d, "label": label}
    for metric, field in (("auc", "AUC"), ("logloss", "logloss")):
        describe(base, metric, [r[field] for r in rows])
    labels.append(base)
cells = []
for m in MODELS:
    for d in DATASETS:
        rows = [r for r in labels if r["model"] == m and r["dataset"] == d]
        assert len(rows) == COUNTS[d]
        base = {"model": m, "dataset": d, "labels": len(rows)}
        for metric in ("auc", "logloss"):
            describe(base, metric, [statistics.fmean(r[f"{metric}_{s}"] for r in rows) for s in SEEDS])
        cells.append(base)
cells.sort(key=lambda row: -row["auc_range"])
labels.sort(key=lambda row: -row["auc_range"])
write_csv("cell_spreads.csv", cells)
write_csv("label_spreads.csv", labels)
write_csv("verified_label_metrics.csv", long_rows)
write_csv("run_log_audit.csv", audit)
summary = {
    "verified_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "user_scope": "Compare all requested seeds without filtering by GPU layout",
    "paper_scope": "OneTrans listed separately following user-provided AGENTS instructions",
    "run_count": len(runs), "logs_reparsed": 2 * len(runs), "label_seed_rows": len(long_rows),
    "cell_count": len(cells), "label_count": len(labels),
    "macro_auc_threshold": 0.005, "label_auc_threshold": 0.01,
    "macro_auc_flagged": sum(r["auc_range"] >= 0.005 for r in cells),
    "label_auc_flagged": sum(r["auc_range"] >= 0.01 for r in labels),
    "logloss_relative_threshold": 0.10,
    "threshold_status": "Descriptive screening thresholds, not significance tests",
    "sources": sources,
}
(OUT / "audit/verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "sources"}, ensure_ascii=False, indent=2))
