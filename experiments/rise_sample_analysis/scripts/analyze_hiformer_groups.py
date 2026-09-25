"""Matched 20262028 HiFormer A/E group analysis on QK-Video and TAAC-25.

count-train reuses the unchanged streaming train-target counter. analyze accepts
only completed formal exports, verifies shard contents and exact paired sample
identity, and uses user-cluster influence intervals conditional on trained models.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import build_group_features as frequencies
from statistical_metrics import auc_and_influence, binary_logloss_rows, cluster_interval


ROOT = Path(__file__).resolve().parents[3]
DATASETS = {
    "QK_Video_Action": {"labels": ("click", "follow", "like", "share"),
                        "test_rows": 48_529_408, "train_rows": 394_647_318},
    "TencentGR_10M_Action": {"labels": ("is_click", "is_conversion"),
                             "test_rows": 153_157_632, "train_rows": 586_117_494},
}
KEYS = ("part_id", "row_offset", "user_id", "item_id", "seq_len", "effective_length")
CI_METHOD = "user-cluster linearized influence; normal 95%; trained models fixed"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes():
    paths = [Path(__file__), SCRIPTS / "statistical_metrics.py", SCRIPTS / "build_group_features.py"]
    return {str(path.relative_to(ROOT)): frequencies.sha256(path) for path in paths}


def frequency_cuts(counts):
    positive = np.asarray(counts[counts > 0])
    if not len(positive):
        raise ValueError("No positive-frequency training entities")
    low, high = np.quantile(positive, [0.25, 0.75], method="inverted_cdf")
    return int(low), int(high)


def frequency_groups(counts, cuts):
    low, high = cuts
    if low < 1 or high < low or np.any(counts < 0):
        raise ValueError("Invalid train-frequency cuts or counts")
    groups = np.full(len(counts), 3, np.int8)
    groups[counts <= high] = 2
    groups[counts <= low] = 1
    groups[counts == 0] = 0
    return groups, {0: "train_target_unseen", 1: f"positive_le_{low}",
                    2: f"gt_{low}_le_{high}", 3: f"gt_{high}"}


def length_groups(length):
    length = np.asarray(length)
    if np.any((length < 0) | (length > 100)):
        raise ValueError("Effective length must lie in [0, 100]")
    return np.select([length == 0, length <= 20, length <= 50, length < 100],
                     [0, 1, 2, 3], default=4).astype(np.int8), {
                         0: "0", 1: "1-20", 2: "21-50", 3: "51-99", 4: "100"}


def count_train(args):
    dataset_dir = ROOT / "datasets" / args.dataset
    metadata = json.loads((dataset_dir / "meta_data.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frequencies.count_train(SimpleNamespace(
        dataset_dir=dataset_dir, output_dir=args.output_dir, batch_size=args.batch_size,
        limit_rows=None, overwrite=False), metadata)
    manifest_path = args.output_dir / "frequencies/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["complete"] or manifest["train_rows_counted"] != DATASETS[args.dataset]["train_rows"]:
        raise ValueError("Train frequency scan did not cover the accepted full training split")
    cuts = {}
    for entity in ("user", "item"):
        counts = np.load(args.output_dir / f"frequencies/{entity}_train_target_count.npy", mmap_mode="r")
        cuts[entity] = list(frequency_cuts(counts))
    frequencies.write_json(args.output_dir / "complete.json", {
        "dataset": args.dataset, "train_rows": manifest["train_rows_counted"], "complete": True,
        "frequency_manifest_sha256": frequencies.sha256(manifest_path), "cuts": cuts,
        "cut_method": "inverted_cdf q25/q75; distinct IDs with positive train-target frequency",
        "source_sha256": source_hashes(), "finished_utc": datetime.now(timezone.utc).isoformat()})


def validate_frequency_inputs(directory, dataset_id):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dataset_dir = ROOT / "datasets" / dataset_id
    if Path(manifest["dataset_dir"]).resolve() != dataset_dir.resolve():
        raise ValueError("Frequency dataset identity mismatch")
    if not manifest["complete"] or manifest["train_rows_counted"] != DATASETS[dataset_id]["train_rows"]:
        raise ValueError("Complete accepted train frequencies required")
    if frequencies.sha256(dataset_dir / "meta_data.json") != manifest["metadata_sha256"]:
        raise ValueError("Training metadata changed since frequency scan")
    for source in manifest["sources"]:
        if frequencies.fingerprint(source["path"]) != source:
            raise ValueError(f"Training source changed: {source['path']}")
    arrays = {}
    for entity in ("user", "item"):
        name = f"{entity}_train_target_count.npy"
        path = directory / name
        if frequencies.sha256(path) != manifest["arrays"][name]:
            raise ValueError(f"Frequency array changed: {path}")
        arrays[entity] = np.load(path, mmap_mode="r")
        if arrays[entity].ndim != 1 or arrays[entity].dtype != np.uint64:
            raise ValueError("Unexpected frequency array schema")
        if int(arrays[entity].sum()) != manifest["train_rows_counted"]:
            raise ValueError("Frequency sum does not match train rows")
    return arrays, manifest


def validate_formal_export(directory, dataset_id, variant):
    """Validate formal status and provenance before reading any group effects."""
    complete = json.loads((directory / "complete.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    shared = json.loads((directory.parent / "manifest.json").read_text())
    labels = DATASETS[dataset_id]["labels"]
    if complete["smoke_only"] or not complete["metric_check_passed"]:
        raise ValueError("Formal nonsmoke global-metric-validated export required")
    if complete["rows"] != DATASETS[dataset_id]["test_rows"] or complete["variant"] != variant:
        raise ValueError("Export scope or coverage mismatch")
    if shared["dataset"] != dataset_id or shared.get("model") != "HiFormer" or shared["split"] != "test":
        raise ValueError("HiFormer dataset/test export required")
    if tuple(shared["labels"]) != labels or shared["batch_size"] != 16384:
        raise ValueError("Label or batch protocol mismatch")
    if shared.get("logical_world_size", shared["world_size"]) != 2:
        raise ValueError("Historical two-rank sample protocol required")
    if manifest["shared_manifest_sha256"] != canonical_hash(shared):
        raise ValueError("Shared manifest changed")
    if manifest["data_version"] != complete["data_version"]:
        raise ValueError("Export data-version receipt mismatch")
    params = manifest["logged_params"]
    if params["model"] != "HiFormer" or params["dataset_id"] != dataset_id or params["seed"] != 20262028:
        raise ValueError("Only matched HiFormer seed20262028 is accepted")
    if bool(params.get("sisa_enabled", False)) != (variant == "E"):
        raise ValueError("A must be native Baseline and E full RISE")
    if variant == "E" and params.get("sisa_variant", "full") != "full":
        raise ValueError("E must be full RISE")
    if set(complete["metrics"]) != set(labels) or set(complete["historical_deviation"]) != set(labels):
        raise ValueError("Incomplete global metric receipt")
    for label in labels:
        for metric in ("AUC", "logloss"):
            value = complete["historical_deviation"][label][metric]
            if value is None or not np.isfinite(value) or abs(value) > 1e-6:
                raise ValueError("Global metric gate does not meet fixed 1e-6 tolerance")
    paths = sorted(directory.glob("*.parquet"))
    if {path.name for path in paths} != set(complete["sample_identity"]):
        raise ValueError("Prediction shard set differs from completion receipt")
    return paths, complete, manifest, shared


def validate_matching_params(before, after):
    differences = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    allowed = {"model_id", "model_root"} | {key for key in differences if key.startswith("sisa_")}
    if differences - allowed:
        raise ValueError(f"Unmatched Baseline/RISE training protocol: {sorted(differences - allowed)}")


def sample_identity(columns, dataset_id, labels):
    digest = hashlib.sha256((dataset_id + "/test").encode())
    for name in KEYS + tuple(labels):
        digest.update(name.encode())
        digest.update(np.asarray(columns[name], dtype="<i8").tobytes())
    return digest.hexdigest()


def read_shard(path, dataset_id, labels, expected_identity):
    receipt = json.loads(path.with_suffix(".json").read_text())
    if frequencies.sha256(path) != receipt["sha256"] or receipt["identity_sha256"] != expected_identity:
        raise ValueError(f"Prediction file/identity receipt mismatch: {path}")
    table = pq.ParquetFile(path).read(columns=["dataset", "split", *KEYS, *labels,
                                              *(f"{label}_pred" for label in labels)])
    if len(table) != receipt["rows"] or not len(table):
        raise ValueError("Invalid shard size")
    if set(table["dataset"].to_pylist()) != {dataset_id} or set(table["split"].to_pylist()) != {"test"}:
        raise ValueError("Shard dataset/split content mismatch")
    columns = {}
    for name in KEYS + tuple(labels) + tuple(f"{label}_pred" for label in labels):
        if table[name].null_count:
            raise ValueError(f"Null values in {name}")
        columns[name] = table[name].to_numpy(zero_copy_only=False)
    if not np.all(columns["part_id"] == receipt["part_id"]) or not np.array_equal(
            columns["row_offset"], np.arange(receipt["row_start"], receipt["row_start"] + len(table))):
        raise ValueError("Shard physical row identity mismatch")
    if sample_identity(columns, dataset_id, labels) != expected_identity:
        raise ValueError("Recomputed sample/label digest differs")
    if np.any(columns["effective_length"] > np.minimum(columns["seq_len"], 100)):
        raise ValueError("Impossible effective history length")
    for label in labels:
        prediction = columns[f"{label}_pred"]
        if not np.isin(columns[label], [0, 1]).all() or not np.isfinite(prediction).all() or np.any(
                (prediction < 0) | (prediction > 1)):
            raise ValueError("Invalid binary labels or probabilities")
    return columns, receipt


def read_paired_exports(baseline_dir, rise_dir, dataset_id):
    before = validate_formal_export(baseline_dir, dataset_id, "A")
    after = validate_formal_export(rise_dir, dataset_id, "E")
    validate_matching_params(before[2]["logged_params"], after[2]["logged_params"])
    if before[1]["data_version"] != after[1]["data_version"] or before[1]["sample_identity"] != after[1]["sample_identity"]:
        raise ValueError("Baseline/RISE sample population differs")
    labels = DATASETS[dataset_id]["labels"]
    accumulated = {name: [] for name in KEYS + labels}
    accumulated.update({f"{variant}_{label}_pred": [] for variant in ("A", "E") for label in labels})
    endpoints = {}
    rows = 0
    for first, second in zip(before[0], after[0]):
        if first.name != second.name:
            raise ValueError("Prediction shard ordering differs")
        identity = before[1]["sample_identity"][first.name]
        a, receipt = read_shard(first, dataset_id, labels, identity)
        e, _ = read_shard(second, dataset_id, labels, identity)
        part = int(receipt["part_id"])
        if receipt["row_start"] != endpoints.get(part, 0) or receipt["rows"] != 16384:
            raise ValueError("Duplicate/gapped/nonhistorical batch interval")
        endpoints[part] = receipt["row_start"] + receipt["rows"]
        for name in KEYS + labels:
            if not np.array_equal(a[name], e[name]):
                raise ValueError(f"Paired metadata/label mismatch: {name}")
            accumulated[name].append(a[name])
        for label in labels:
            accumulated[f"A_{label}_pred"].append(a[f"{label}_pred"])
            accumulated[f"E_{label}_pred"].append(e[f"{label}_pred"])
        rows += len(a["row_offset"])
    expected = {int(part): int(count) for part, count in before[3]["expected_rows_by_part"].items()}
    if endpoints != expected or rows != before[1]["rows"]:
        raise ValueError("Retained source coverage is not complete")
    data = {name: np.concatenate(chunks) for name, chunks in accumulated.items()}
    provenance = {"A": before[1], "E": after[1], "manifests": {
        "baseline": frequencies.sha256(baseline_dir / "manifest.json"),
        "rise": frequencies.sha256(rise_dir / "manifest.json")}}
    return data, provenance


def interval_fields(estimate, influence, prefix):
    if influence is None or len(influence) < 2:
        return {f"{prefix}_cluster_se": None, f"{prefix}_ci_lower": None, f"{prefix}_ci_upper": None}
    result = cluster_interval(estimate, influence)
    return {f"{prefix}_cluster_se": result["cluster_se"],
            f"{prefix}_ci_lower": result["ci_lower"], f"{prefix}_ci_upper": result["ci_upper"]}


def summarize_label(y, before, after, users, user_mapping=None):
    y = np.asarray(y)
    if not len(y):
        raise ValueError("Empty groups must be recorded in definitions, not estimated")
    values, inverse = np.unique(users, return_inverse=True) if user_mapping is None else user_mapping
    size = len(y)
    if len(inverse) != size:
        raise ValueError("User mapping does not align with group rows")
    positive = int(y.sum())
    # Official evaluation casts probabilities to float64 before logloss clipping.
    a_loss = binary_logloss_rows(y, np.asarray(before, dtype=np.float64))
    e_loss = binary_logloss_rows(y, np.asarray(after, dtype=np.float64))
    delta_loss_rows = e_loss - a_loss
    delta_loss = float(delta_loss_rows.mean())
    ll_cluster = np.bincount(inverse, weights=(delta_loss_rows - delta_loss) / size, minlength=len(values))
    if 0 < positive < size:
        a_auc, a_if = auc_and_influence(y, before)
        e_auc, e_if = auc_and_influence(y, after)
        delta_auc = e_auc - a_auc
        auc_cluster = np.bincount(inverse, weights=e_if - a_if, minlength=len(values))
    else:
        a_auc = e_auc = delta_auc = None
        auc_cluster = None
    row = {"comparison": "E-A", "rows": size, "users": len(values), "positives": positive,
           "negatives": size - positive, "positive_fraction": positive / size,
           "baseline_auc": a_auc, "rise_auc": e_auc, "delta_auc": delta_auc,
           "baseline_logloss": float(a_loss.mean()), "rise_logloss": float(e_loss.mean()),
           "delta_logloss": delta_loss, "macro_labels": 1,
           **interval_fields(delta_auc, auc_cluster, "auc"),
           **interval_fields(delta_loss, ll_cluster, "logloss")}
    return row, {"users": values, "auc": auc_cluster, "logloss": ll_cluster}


def summarize_macro(results):
    if not results:
        raise ValueError("No label estimates")
    users = results[0][1]["users"]
    if not all(np.array_equal(users, payload["users"]) for _, payload in results):
        raise ValueError("Macro labels must use identical user populations")
    rows = [row for row, _ in results]
    if not all(row["rows"] == rows[0]["rows"] for row in rows):
        raise ValueError("Macro labels must use identical sample populations")
    all_auc_defined = all(row["delta_auc"] is not None for row in rows)
    macro = {"comparison": "E-A", "rows": rows[0]["rows"], "users": len(users),
             "positives": None, "negatives": None, "positive_fraction": None, "macro_labels": len(rows)}
    for name in ("baseline_auc", "rise_auc", "delta_auc"):
        macro[name] = float(np.mean([row[name] for row in rows])) if all_auc_defined else None
    for name in ("baseline_logloss", "rise_logloss", "delta_logloss"):
        macro[name] = float(np.mean([row[name] for row in rows]))
    # Average each user's label contributions BEFORE estimating their variance.
    auc_cluster = np.mean([payload["auc"] for _, payload in results], axis=0) if all_auc_defined else None
    ll_cluster = np.mean([payload["logloss"] for _, payload in results], axis=0)
    macro.update(interval_fields(macro["delta_auc"], auc_cluster, "auc"))
    macro.update(interval_fields(macro["delta_logloss"], ll_cluster, "logloss"))
    return macro


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data, provenance = read_paired_exports(args.baseline_dir, args.rise_dir, args.dataset)
    arrays, frequency_manifest = validate_frequency_inputs(args.frequency_dir, args.dataset)
    cuts = {entity: frequency_cuts(counts) for entity, counts in arrays.items()}
    group_definitions = {}
    rows = []
    for dimension in args.dimensions:
        if dimension == "effective_length":
            membership, names = length_groups(data["effective_length"])
            description = {"definition": "Valid input history tokens, excludes target/padding; max_len100"}
        else:
            entity = dimension.split("_")[0]
            identifiers = data[f"{entity}_id"].astype(np.int64, copy=False)
            frequencies.validate_ids(identifiers, len(arrays[entity]), f"test {entity}")
            membership, names = frequency_groups(arrays[entity][identifiers], cuts[entity])
            description = {"cuts": list(cuts[entity]), "quantile_method": "inverted_cdf",
                           "cut_population": "Distinct encoded global IDs with positive training-target frequency",
                           "zero_definition": "No training target occurrence; not necessarily absent from training histories"}
        group_definitions[dimension] = {"groups": names, **description, "empty_groups": []}
        for group, name in names.items():
            selected = membership == group
            if not np.any(selected):
                group_definitions[dimension]["empty_groups"].append(name)
                continue
            users = data["user_id"][selected]
            mapping = np.unique(users, return_inverse=True)
            estimates = []
            for label in DATASETS[args.dataset]["labels"]:
                result = summarize_label(data[label][selected], data[f"A_{label}_pred"][selected],
                                         data[f"E_{label}_pred"][selected], users, mapping)
                estimates.append(result)
                rows.append({"dataset": args.dataset, "dimension": dimension, "group": name,
                             "label": label, **result[0]})
            rows.append({"dataset": args.dataset, "dimension": dimension, "group": name,
                         "label": f"macro_{len(estimates)}_tasks", **summarize_macro(estimates)})
            print(json.dumps({"dimension": dimension, "group": name, "rows": int(selected.sum())}), flush=True)
            del estimates, mapping, users
        write_csv(args.output_dir / "group_results.csv", rows)
    manifest = {"dataset": args.dataset, "model": "HiFormer", "seed": 20262028, "variants": ["A", "E"],
                "rows": len(data["row_offset"]), "labels": list(DATASETS[args.dataset]["labels"]),
                "definitions": group_definitions, "source_sha256": source_hashes(),
                "prediction_receipts": {variant: {key: value for key, value in provenance[variant].items()
                                                   if key != "sample_identity"} for variant in ("A", "E")},
                "prediction_manifests_sha256": provenance["manifests"],
                "frequency_manifest_sha256": frequencies.sha256(args.frequency_dir / "manifest.json"),
                "frequency_train_rows": frequency_manifest["train_rows_counted"], "ci_method": CI_METHOD,
                "macro_variance": "Average label influences within each global user, then cluster sandwich; never average SEs",
                "statistical_scope": "Exploratory conditional effects; no training-seed variance or multiplicity-adjusted claims",
                "created_utc": datetime.now(timezone.utc).isoformat()}
    frequencies.write_json(args.output_dir / "manifest.json", manifest)
    frequencies.write_json(args.output_dir / "complete.json", {
        "dataset": args.dataset, "rows": len(data["row_offset"]), "result_rows": len(rows),
        "identity_verified": True, "variants": ["A", "E"],
        "group_results_sha256": frequencies.sha256(args.output_dir / "group_results.csv"),
        "manifest_sha256": frequencies.sha256(args.output_dir / "manifest.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("count-train", "analyze"):
        command = subparsers.add_parser(name)
        command.add_argument("--dataset", choices=list(DATASETS), required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "count-train":
            command.add_argument("--batch-size", type=int, default=262144)
        else:
            command.add_argument("--baseline-dir", type=Path, required=True)
            command.add_argument("--rise-dir", type=Path, required=True)
            command.add_argument("--frequency-dir", type=Path, required=True)
            command.add_argument("--dimensions", nargs="+", choices=["effective_length", "user_train_target_frequency",
                                  "item_train_target_frequency"], default=["effective_length", "user_train_target_frequency",
                                                                         "item_train_target_frequency"])
    args = parser.parse_args()
    cpus = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    pa.set_cpu_count(cpus)
    pa.set_io_thread_count(cpus)
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT / "artifacts/rise_sample_analysis"):
        parser.error("Output must be in the isolated artifacts/rise_sample_analysis subtree")
    if args.command == "count-train":
        if args.batch_size < 1:
            parser.error("Positive batch size required")
        count_train(args)
    else:
        if len(set(args.dimensions)) != len(args.dimensions):
            parser.error("Duplicate analysis dimensions")
        analyze(args)


if __name__ == "__main__":
    main()
