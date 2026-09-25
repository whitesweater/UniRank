"""Additional probability/ranking quality for formally paired native A and RISE E.

AP is sklearn.average_precision_score (non-interpolated average precision), not
trapezoidal PR-AUC. No calibrator or threshold is fitted. The shared hybrid
probability grid below is fixed before this analysis and used for every label.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from statistical_metrics import binary_logloss_rows

ROOT = Path(__file__).resolve().parents[3]
DATASETS = {
    "MerRec_Action": {"model": "HyFormer", "labels": ("Like", "Cart", "Offer", "Checkout", "Purchase"), "rows": 11_599_872},
    "QK_Video_Action": {"model": "HiFormer", "labels": ("click", "follow", "like", "share"), "rows": 48_529_408},
    "TencentGR_10M_Action": {"model": "HiFormer", "labels": ("is_click", "is_conversion"), "rows": 153_157_632},
}
KEYS = ("part_id", "row_offset", "user_id", "item_id", "seq_len", "effective_length")
BIN_EDGES = np.array([0., 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, .001, .003, .01, .03,
                      .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.], dtype=np.float64)
METRICS = ("ROC_AUC", "AP", "Brier", "Logloss", "ECE_fixed_grid")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def source_record(path):
    path = Path(path)
    return {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def sample_digest(columns, dataset_id, labels):
    digest = hashlib.sha256((dataset_id + "/test").encode())
    for name in KEYS + tuple(labels):
        digest.update(name.encode())
        digest.update(np.asarray(columns[name], dtype="<i8").tobytes())
    return digest.hexdigest()


def inspect_export(directory, dataset_id, variant):
    spec = DATASETS[dataset_id]
    shared = json.loads((directory.parent / "manifest.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    complete = json.loads((directory / "complete.json").read_text())
    if complete["smoke_only"] or not complete["metric_check_passed"] or complete["variant"] != variant:
        raise ValueError("Formal validated A/E predictions required")
    if complete["rows"] != spec["rows"] or shared["dataset"] != dataset_id or shared["split"] != "test":
        raise ValueError("Dataset or retained test coverage mismatch")
    if tuple(shared["labels"]) != spec["labels"] or shared["batch_size"] != 16384:
        raise ValueError("Historical label/batch protocol mismatch")
    if shared.get("logical_world_size", shared.get("world_size")) != 2:
        raise ValueError("The historical two-rank sample population is required")
    if canonical_hash(shared) != manifest["shared_manifest_sha256"] or manifest["data_version"] != complete["data_version"]:
        raise ValueError("Export manifest integrity mismatch")
    params = manifest["logged_params"]
    if params["model"] != spec["model"] or params["dataset_id"] != dataset_id or params["seed"] != 20262028:
        raise ValueError("Only this matched seed20262028 comparison is accepted")
    if bool(params.get("sisa_enabled", False)) != (variant == "E"):
        raise ValueError("A must be a native baseline and E must enable RISE")
    if variant == "E" and params.get("sisa_variant", "full") != "full":
        raise ValueError("E must be full RISE")
    if set(complete["metrics"]) != set(spec["labels"]) or set(complete["historical_deviation"]) != set(spec["labels"]):
        raise ValueError("Incomplete global metric receipt")
    for label in spec["labels"]:
        for name in ("AUC", "logloss"):
            delta = complete["historical_deviation"][label][name]
            if delta is None or not np.isfinite(delta) or abs(delta) > 1e-6:
                raise ValueError("Historical global metric check did not meet 1e-6")
    paths = sorted(directory.glob("*.parquet"))
    if {path.name for path in paths} != set(complete["sample_identity"]):
        raise ValueError("Prediction shards differ from completion receipt")
    return {"shared": shared, "manifest": manifest, "complete": complete, "paths": paths}


def read_shard(path, dataset_id, labels, expected_identity):
    receipt = json.loads(path.with_suffix(".json").read_text())
    digest = sha256(path)
    if digest != receipt["sha256"] or receipt["identity_sha256"] != expected_identity:
        raise ValueError("Shard SHA or identity receipt mismatch")
    table = pq.ParquetFile(path).read(columns=["dataset", "split", *KEYS, *labels,
                                              *(f"{label}_pred" for label in labels)])
    if len(table) != receipt["rows"] or len(table) != 16384:
        raise ValueError("Unexpected historical batch size")
    if set(table["dataset"].to_pylist()) != {dataset_id} or set(table["split"].to_pylist()) != {"test"}:
        raise ValueError("Wrong dataset/split inside prediction shard")
    columns = {}
    for name in KEYS + tuple(labels) + tuple(f"{label}_pred" for label in labels):
        if table[name].null_count:
            raise ValueError(f"Null values in {name}")
        columns[name] = table[name].to_numpy(zero_copy_only=False)
    if not np.all(columns["part_id"] == receipt["part_id"]) or not np.array_equal(
            columns["row_offset"], np.arange(receipt["row_start"], receipt["row_start"] + len(table))):
        raise ValueError("Physical row ID mismatch")
    if sample_digest(columns, dataset_id, labels) != expected_identity:
        raise ValueError("Actual row/label metadata differ from the frozen identity")
    return columns, receipt, {"path": relative(path), "bytes": path.stat().st_size, "sha256": digest}


def read_pair(baseline_dir, rise_dir, dataset_id):
    labels = DATASETS[dataset_id]["labels"]
    exports = {"A": inspect_export(baseline_dir, dataset_id, "A"), "E": inspect_export(rise_dir, dataset_id, "E")}
    a, e = exports["A"], exports["E"]
    if a["complete"]["data_version"] != e["complete"]["data_version"] or a["complete"]["sample_identity"] != e["complete"]["sample_identity"]:
        raise ValueError("A/E do not have the same sample population")
    before, after = a["manifest"]["logged_params"], e["manifest"]["logged_params"]
    differences = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    if differences - {"model_id", "model_root"} - {key for key in differences if key.startswith("sisa_")}:
        raise ValueError("A/E training protocol mismatch")
    chunks = {label: [] for label in labels}
    chunks.update({f"{variant}_{label}": [] for variant in ("A", "E") for label in labels})
    endpoints, sources, total = {}, [], 0
    for directory in (baseline_dir, rise_dir):
        sources.extend(source_record(path) for path in (directory.parent / "manifest.json", directory / "manifest.json", directory / "complete.json"))
    for first, second in zip(a["paths"], e["paths"]):
        if first.name != second.name:
            raise ValueError("A/E shard ordering differs")
        identity = a["complete"]["sample_identity"][first.name]
        left, receipt, left_source = read_shard(first, dataset_id, labels, identity)
        right, _, right_source = read_shard(second, dataset_id, labels, identity)
        sources.extend([left_source, right_source])
        part = int(receipt["part_id"])
        if endpoints.get(part, 0) != receipt["row_start"]:
            raise ValueError("Duplicate or missing original row interval")
        endpoints[part] = receipt["row_start"] + receipt["rows"]
        for name in KEYS + labels:
            if not np.array_equal(left[name], right[name]):
                raise ValueError(f"Paired sample mismatch in {name}")
        for label in labels:
            chunks[label].append(left[label])
            chunks[f"A_{label}"].append(left[f"{label}_pred"])
            chunks[f"E_{label}"].append(right[f"{label}_pred"])
        total += receipt["rows"]
    expected = {int(part): int(count) for part, count in a["shared"]["expected_rows_by_part"].items()}
    if endpoints != expected or total != a["complete"]["rows"]:
        raise ValueError("Incomplete retained test population")
    return {name: np.concatenate(parts) for name, parts in chunks.items()}, exports, sources


def validate_probabilities(y, scores):
    y = np.asarray(y)
    p = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Nonempty aligned one-dimensional labels/probabilities required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Labels must be binary and probabilities finite in [0,1]")
    return y.astype(np.uint8, copy=False), p


def calibration_table(y, scores, edges=BIN_EDGES):
    y, p = validate_probabilities(y, scores)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or edges[0] != 0 or edges[-1] != 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("Calibration edges must strictly increase from zero to one")
    membership = np.minimum(np.searchsorted(edges, p, side="right") - 1, len(edges) - 2)
    n = np.bincount(membership, minlength=len(edges) - 1)
    positive = np.bincount(membership, weights=y, minlength=len(edges) - 1)
    total_prediction = np.bincount(membership, weights=p, minlength=len(edges) - 1)
    rows = []
    for index, size in enumerate(n):
        observed = float(positive[index] / size) if size else None
        predicted = float(total_prediction[index] / size) if size else None
        rows.append({"bin_id": index, "lower": float(edges[index]), "upper": float(edges[index + 1]),
                     "closure": "[lower,upper]" if index == len(n) - 1 else "[lower,upper)",
                     "N": int(size), "positive_count": int(positive[index]), "positive_rate": observed,
                     "mean_prediction": predicted, "population_fraction": float(size / len(y)),
                     "empty": bool(size == 0)})
    return rows


def quality_metrics(y, scores):
    y, p = validate_probabilities(y, scores)
    positive = int(y.sum())
    bins = calibration_table(y, p)
    ece = sum(row["population_fraction"] * abs(row["positive_rate"] - row["mean_prediction"])
              for row in bins if not row["empty"])
    return {"N": len(y), "positive_count": positive, "negative_count": len(y) - positive,
            "positive_rate": positive / len(y), "mean_prediction": float(p.mean()),
            "ROC_AUC": float(roc_auc_score(y, p)) if 0 < positive < len(y) else None,
            "AP": float(average_precision_score(y, p)) if positive else None,
            "Brier": float(np.mean((p - y) ** 2)),
            "Logloss": float(binary_logloss_rows(y, p).mean()), "ECE_fixed_grid": float(ece)}, bins


def compare_metrics(before, after):
    result = {}
    for name in METRICS:
        result[f"Baseline_{name}"] = before[name]
        result[f"RISE_{name}"] = after[name]
        result[f"delta_{name}"] = None if before[name] is None or after[name] is None else after[name] - before[name]
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value, signed=False):
    if value is None:
        return "不可定义"
    return format(value, "+.8f" if signed else ".8f")


def write_readout(path, dataset_id, deltas, count):
    lines = [f"# {DATASETS[dataset_id]['model']}–{dataset_id}：预测质量补充分析", "",
             f"固定 seed20262028 的原生 Baseline（A）与完整 RISE（E），在相同 {count:,} 条正式测试样本上比较。未重新训练、未拟合校准器、未从测试集选择阈值。", "",
             "AP 使用 sklearn.average_precision_score，即非插值平均精确率，不是梯形积分的 PR-AUC。Brier 为未裁剪概率的均方误差；Logloss 先将概率转为 float64，再按机器精度裁剪，与正式评估口径一致。", "",
             "| 动作 | Baseline AP | RISE AP | ΔAP | Baseline Brier | RISE Brier | ΔBrier | Baseline Logloss | RISE Logloss | ΔLogloss |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in deltas:
        values = [row["label"]]
        for metric in ("AP", "Brier", "Logloss"):
            values.extend([format_value(row[f"Baseline_{metric}"]), format_value(row[f"RISE_{metric}"]),
                           format_value(row[f"delta_{metric}"], True)])
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "差值均为 RISE 减 Baseline；AP 越高越好，Brier 与 Logloss 越低越好。宏平均对所有动作等权，不混合不同任务的样本。ROC-AUC、AP和概率损失刻画不同方面，改善方向不必一致。", "",
              "校准网格在本次分析执行前固定，对两个模型和所有标签相同：" + ", ".join(f"{edge:g}" for edge in BIN_EDGES) + "。每箱左闭右开，最后一箱包含1；空箱仍列出N=0，正例比例和平均预测为空。完整结果见 calibration_bins.csv。", "",
              "ECE_fixed_grid 仅是上述固定分箱的描述统计，依赖边界选择；不视为与分箱无关的校准质量结论。未使用任何输出反向选择分箱。", "",
              "本分析是单种子、固定已训练模型的离线描述，没有新增训练重复或统计显著性声明，也不推断线上效用。现有预处理的事件归因时点及跨划分边界仍未解决；本分析沿用该数据，不证明这些问题不存在。", ""]
    path.write_text("\n".join(lines))


def analyze(dataset_id, baseline_dir, rise_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    code_paths = (Path(__file__).resolve(), SCRIPTS / "statistical_metrics.py")
    code_hashes = {relative(path): sha256(path) for path in code_paths}
    data, exports, sources = read_pair(baseline_dir, rise_dir, dataset_id)
    metric_rows, bin_rows, deltas, global_rechecks = [], [], [], []
    maps = {"A": {}, "E": {}}
    for label in DATASETS[dataset_id]["labels"]:
        for variant, name in (("A", "Baseline"), ("E", "RISE")):
            metrics, bins = quality_metrics(data[label], data[f"{variant}_{label}"])
            saved = exports[variant]["complete"]["metrics"][label]
            recheck = {"variant": variant, "label": label,
                       "AUC_deviation_from_export": metrics["ROC_AUC"] - saved["AUC"],
                       "Logloss_deviation_from_export": metrics["Logloss"] - saved["logloss"]}
            if max(abs(recheck["AUC_deviation_from_export"]), abs(recheck["Logloss_deviation_from_export"])) > 1e-10:
                raise ValueError(f"Independent official metric recheck failed: {recheck}")
            global_rechecks.append(recheck)
            maps[variant][label] = metrics
            metric_rows.append({"dataset": dataset_id, "label": label, "variant": variant, "name": name, **metrics})
            bin_rows.extend({"dataset": dataset_id, "label": label, "variant": variant, "name": name, **row} for row in bins)
        deltas.append({"dataset": dataset_id, "label": label, "N": len(data[label]),
                       "positive_count": int(data[label].sum()), "positive_rate": float(data[label].mean()),
                       **compare_metrics(maps["A"][label], maps["E"][label])})
        print(json.dumps({"label": label, "Baseline_AP": maps["A"][label]["AP"], "RISE_AP": maps["E"][label]["AP"]}), flush=True)
        gc.collect()
    macro = {variant: {metric: float(np.mean([values[metric] for values in maps[variant].values()]))
                       for metric in METRICS} for variant in ("A", "E")}
    deltas.append({"dataset": dataset_id, "label": f"macro_{len(maps['A'])}_tasks", "N": DATASETS[dataset_id]["rows"],
                   "positive_count": None, "positive_rate": None, **compare_metrics(macro["A"], macro["E"])})
    write_csv(output_dir / "label_metrics.csv", metric_rows)
    write_csv(output_dir / "paired_deltas.csv", deltas)
    write_csv(output_dir / "calibration_bins.csv", bin_rows)
    write_csv(output_dir / "input_sources.csv", sources)
    write_readout(output_dir / "summary.md", dataset_id, deltas, DATASETS[dataset_id]["rows"])
    if code_hashes != {relative(path): sha256(path) for path in code_paths}:
        raise ValueError("Analysis source changed during execution")
    manifest = {"dataset": dataset_id, "model": DATASETS[dataset_id]["model"], "seed": 20262028,
                "variants": ["A", "E"], "rows": DATASETS[dataset_id]["rows"], "labels": list(DATASETS[dataset_id]["labels"]),
                "paired_sample_identity_verified": True, "source_sha256": code_hashes,
                "sklearn_version": sklearn.__version__, "AP_definition": "sklearn.metrics.average_precision_score; non-interpolated; not trapezoidal PR-AUC",
                "Brier_definition": "Mean squared error of original probabilities; no clipping",
                "Logloss_definition": "Float64 conversion then epsilon(float64) clipping, matching official evaluation",
                "calibration_edges": BIN_EDGES.tolist(), "calibration_closure": "Left closed/right open; final bin includes 1",
                "empty_bin_policy": "N=0, positive_count=0; positive_rate and mean_prediction null",
                "bin_policy": "Same fixed generic log/linear probability grid for every task and variant; chosen before this quality run, never fitted to test output",
                "ECE_scope": "Bin-dependent descriptive weighted absolute bin calibration error only",
                "calibrator_fitted": False, "threshold_selected": False, "training_executed": False,
                "uncertainty_scope": "Single seed and fixed models; descriptive estimates without new confidence intervals or online claims",
                "event_attribution_scope": "Inherited preprocessing event-time and split-attribution questions remain unresolved",
                "global_metric_rechecks": global_rechecks, "independent_recheck_tolerance": 1e-10,
                "prediction_completion_sha256": {variant: sha256(directory / "complete.json")
                                                 for variant, directory in (("A", baseline_dir), ("E", rise_dir))},
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "finished_utc": datetime.now(timezone.utc).isoformat()}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    outputs = {path.name: sha256(path) for path in output_dir.iterdir() if path.is_file()}
    (output_dir / "complete.json").write_text(json.dumps({"rows": manifest["rows"], "labels": manifest["labels"],
        "paired_sample_identity_verified": True, "global_metric_recheck_passed": True, "outputs_sha256": outputs}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), default="MerRec_Action")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--rise-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cpus = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    pa.set_cpu_count(cpus)
    pa.set_io_thread_count(cpus)
    output = args.output_dir.resolve()
    if not any(output.is_relative_to(path) for path in (ROOT / "experiments/rise_sample_analysis/results",
                                                       ROOT / "artifacts/rise_sample_analysis")):
        parser.error("Output must be inside this isolated study's results or artifacts")
    analyze(args.dataset, args.baseline_dir.resolve(), args.rise_dir.resolve(), output)


if __name__ == "__main__":
    main()
