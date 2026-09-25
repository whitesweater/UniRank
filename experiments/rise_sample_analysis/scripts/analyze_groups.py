"""Analyze paired MerRec predictions with frozen input-defined sample groups.

Only completed, nonsmoke exports with passed global-metric gates are accepted.
Intervals use user-cluster aggregated empirical AUC influence functions; they
condition on trained models and do not estimate training-seed variability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq

from statistical_metrics import auc_and_influence, binary_logloss_rows, cluster_interval


ROOT = Path(__file__).resolve().parents[3]
LABELS = ("Like", "Cart", "Offer", "Checkout", "Purchase")
KEYS = ("part_id", "row_offset", "user_id", "item_id", "seq_len", "effective_length")
PAIRS = (("E", "B"), ("C", "B"), ("D", "B"), ("E", "C"), ("E", "D"), ("E", "A"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_columns(paths, columns):
    arrays = {column: [] for column in columns}
    for path in paths:
        table = pq.ParquetFile(path).read(columns=columns)
        for column in columns:
            arrays[column].append(table[column].to_numpy(zero_copy_only=False))
    return {column: np.concatenate(parts) for column, parts in arrays.items()}


def load_predictions(directory):
    complete = json.loads((directory / "complete.json").read_text())
    if complete["smoke_only"] or not complete["metric_check_passed"]:
        raise ValueError(f"Formal metric-validated predictions required: {directory}")
    paths = sorted(directory.glob("*.parquet"))
    if set(p.name for p in paths) != set(complete["sample_identity"]):
        raise ValueError("Prediction shard set differs from receipt")
    for path in paths:
        receipt = json.loads(path.with_suffix(".json").read_text())
        if sha256(path) != receipt["sha256"] or receipt["identity_sha256"] != complete["sample_identity"][path.name]:
            raise ValueError(f"Prediction receipt mismatch: {path}")
    data = load_columns(paths, list(KEYS) + list(LABELS) + [label + "_pred" for label in LABELS])
    if len(data["row_offset"]) != complete["rows"]:
        raise ValueError("Prediction rows differ from completion receipt")
    return data, complete


def unique_sorted_keys(data):
    parts, offsets = data["part_id"], data["row_offset"]
    if np.any(parts < 0) or np.any(parts >= 2**15) or np.any(offsets < 0) or np.any(offsets >= 2**48):
        raise ValueError("Sample identity outside safe encoding range")
    key = (parts.astype(np.int64) << 48) | offsets.astype(np.int64)
    if len(key) > 1 and np.any(key[1:] <= key[:-1]):
        raise ValueError("Rows must have unique, increasing physical identities")
    return key


def frequency_cuts(path):
    counts = np.load(path, mmap_mode="r")
    positive = np.asarray(counts[counts > 0])
    if not len(positive):
        raise ValueError("No observed training entities")
    low, high = np.quantile(positive, [0.25, 0.75], method="inverted_cdf")
    return int(low), int(high)


def frequency_groups(counts, cuts):
    low, high = cuts
    group = np.full(len(counts), 3, dtype=np.int8)
    group[counts <= high] = 2
    group[counts <= low] = 1
    group[counts == 0] = 0
    return group, {0: "train_target_unseen", 1: f"positive_le_{low}",
                   2: f"gt_{low}_le_{high}", 3: f"gt_{high}"}


def dimensions(features, frequency_directory):
    length = features["effective_length"]
    length_group = np.select([length == 0, length <= 20, length <= 50, length < 100],
                             [0, 1, 2, 3], default=4).astype(np.int8)
    yield "effective_length", length_group, {0: "0", 1: "1-20", 2: "21-50", 3: "51-99", 4: "100"}, {}
    for entity in ("item", "user"):
        cuts = frequency_cuts(frequency_directory / f"{entity}_train_target_count.npy")
        group, names = frequency_groups(features[f"{entity}_train_target_count"], cuts)
        yield f"{entity}_train_target_frequency", group, names, {
            "positive_entity_quartiles": list(cuts), "cut_population": "distinct training entities with positive target frequency"}
    entropy = features["c1_entropy"]
    entropy_groups = np.full(len(length), -1, dtype=np.int8)
    selected = length == 100
    entropy_groups[selected] = np.select(
        [~np.isfinite(entropy[selected]), entropy[selected] <= 1e-7,
         entropy[selected] <= np.log(2), entropy[selected] <= np.log(5)], [0, 1, 2, 3], default=4)
    yield "c1_entropy_at_length100", entropy_groups, {
        0: "undefined", 1: "one_effective_category", 2: "effective_categories_1_to_2",
        3: "effective_categories_2_to_5", 4: "effective_categories_gt5"}, {
            "restricted_to_effective_length": 100, "excluded_category_codes": [0, 309]}
    fraction = features["target_c1_history_fraction"]
    consistency = np.full(len(length), -1, dtype=np.int8)
    consistency[selected] = np.select([
        ~np.isfinite(fraction[selected]), fraction[selected] == 0,
        fraction[selected] <= 0.25, fraction[selected] <= 0.75], [0, 1, 2, 3], default=4)
    yield "candidate_category_support_at_length100", consistency, {
        0: "undefined", 1: "zero", 2: "positive_le_0.25", 3: "gt_0.25_le_0.75", 4: "gt_0.75"}, {
            "restricted_to_effective_length": 100, "excluded_category_codes": [0, 309]}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def summarize_group(y, predictions, users):
    """All variants share one user mapping and one group population."""
    n = len(y); positive = int(y.sum()); negative = n - positive
    user_values, user_inverse = np.unique(users, return_inverse=True)
    count = len(user_values)
    points, clusters, loss_clusters = {}, {}, {}
    for variant, scores in predictions.items():
        # Repository evaluation first promotes predictions to float64, then
        # sklearn applies dtype-specific clipping. Preserve that edge behavior.
        losses = binary_logloss_rows(y, np.asarray(scores, dtype=np.float64))
        mean_loss = float(losses.mean())
        loss_clusters[variant] = np.bincount(user_inverse, weights=(losses - mean_loss) / n, minlength=count)
        if positive and negative:
            auc, influence = auc_and_influence(y, scores)
            clusters[variant] = np.bincount(user_inverse, weights=influence, minlength=count)
        else:
            auc = None
        points[variant] = {"variant": variant, "rows": n, "users": count, "positives": positive,
                           "negatives": negative, "positive_fraction": positive / n,
                           "AUC": auc, "logloss": mean_loss}
    effects = []
    for treatment, control in PAIRS:
        if treatment not in points or control not in points:
            continue
        record = {"comparison": f"{treatment}-{control}", "rows": n, "users": count,
                  "positives": positive, "negatives": negative, "positive_fraction": positive / n,
                  "control_auc": points[control]["AUC"], "treatment_auc": points[treatment]["AUC"]}
        if positive and negative and count > 1:
            estimate = points[treatment]["AUC"] - points[control]["AUC"]
            interval = cluster_interval(estimate, clusters[treatment] - clusters[control])
            record.update(delta_auc=estimate, auc_cluster_se=interval["cluster_se"],
                          auc_ci_lower=interval["ci_lower"], auc_ci_upper=interval["ci_upper"])
        else:
            record.update(delta_auc=None, auc_cluster_se=None, auc_ci_lower=None, auc_ci_upper=None)
        delta_loss = points[treatment]["logloss"] - points[control]["logloss"]
        if count > 1:
            interval = cluster_interval(delta_loss, loss_clusters[treatment] - loss_clusters[control])
            record.update(delta_logloss=delta_loss, logloss_cluster_se=interval["cluster_se"],
                          logloss_ci_lower=interval["ci_lower"], logloss_ci_upper=interval["ci_upper"])
        else:
            record.update(delta_logloss=delta_loss, logloss_cluster_se=None,
                          logloss_ci_lower=None, logloss_ci_upper=None)
        effects.append(record)
    if all(variant in points for variant in "BCDE"):
        if positive and negative and count > 1:
            value = points["E"]["AUC"] - points["C"]["AUC"] - points["D"]["AUC"] + points["B"]["AUC"]
            interval = cluster_interval(value, clusters["E"] - clusters["C"] - clusters["D"] + clusters["B"])
            effects.append({"comparison": "E-C-D+B", "rows": n, "users": count,
                            "positives": positive, "negatives": negative, "positive_fraction": positive / n,
                            "delta_auc": value, "auc_cluster_se": interval["cluster_se"],
                            "auc_ci_lower": interval["ci_lower"], "auc_ci_upper": interval["ci_upper"]})
    return list(points.values()), effects, (user_values, clusters, loss_clusters)


def macro_summary(label_results):
    """Average label effects before computing variance, keeping task covariance."""
    first_points, _, first_payload = label_results[0]
    users = first_payload[0]
    if not all(np.array_equal(users, result[2][0]) for result in label_results):
        raise ValueError("Macro labels must use the identical sample/user population")
    variants = [point["variant"] for point in first_points]
    point_maps = [{p["variant"]: p for p in result[0]} for result in label_results]
    points, auc_clusters, loss_clusters = {}, {}, {}
    n, k = first_points[0]["rows"], len(users)
    for variant in variants:
        aucs = [values[variant]["AUC"] for values in point_maps]
        auc = None if any(value is None for value in aucs) else float(np.mean(aucs))
        points[variant] = {"variant": variant, "rows": n, "users": k,
                           "positives": None, "negatives": None, "positive_fraction": None,
                           "AUC": auc, "logloss": float(np.mean([m[variant]["logloss"] for m in point_maps])),
                           "macro_labels": len(label_results)}
        if auc is not None:
            auc_clusters[variant] = np.mean([result[2][1][variant] for result in label_results], axis=0)
        loss_clusters[variant] = np.mean([result[2][2][variant] for result in label_results], axis=0)
    effects = []
    for treatment, control in PAIRS:
        if treatment not in points or control not in points:
            continue
        row = {"comparison": f"{treatment}-{control}", "rows": n, "users": k,
               "positives": None, "negatives": None, "positive_fraction": None,
               "macro_labels": len(label_results), "control_auc": points[control]["AUC"],
               "treatment_auc": points[treatment]["AUC"]}
        if treatment in auc_clusters and control in auc_clusters and k > 1:
            value = points[treatment]["AUC"] - points[control]["AUC"]
            interval = cluster_interval(value, auc_clusters[treatment] - auc_clusters[control])
            row.update(delta_auc=value, auc_cluster_se=interval["cluster_se"],
                       auc_ci_lower=interval["ci_lower"], auc_ci_upper=interval["ci_upper"])
        else:
            row.update(delta_auc=None, auc_cluster_se=None, auc_ci_lower=None, auc_ci_upper=None)
        value = points[treatment]["logloss"] - points[control]["logloss"]
        if k > 1:
            interval = cluster_interval(value, loss_clusters[treatment] - loss_clusters[control])
            row.update(delta_logloss=value, logloss_cluster_se=interval["cluster_se"],
                       logloss_ci_lower=interval["ci_lower"], logloss_ci_upper=interval["ci_upper"])
        else:
            row.update(delta_logloss=value, logloss_cluster_se=None,
                       logloss_ci_lower=None, logloss_ci_upper=None)
        effects.append(row)
    if all(v in auc_clusters for v in "BCDE") and k > 1:
        value = points["E"]["AUC"] - points["C"]["AUC"] - points["D"]["AUC"] + points["B"]["AUC"]
        interval = cluster_interval(value, auc_clusters["E"] - auc_clusters["C"] - auc_clusters["D"] + auc_clusters["B"])
        effects.append({"comparison": "E-C-D+B", "rows": n, "users": k,
                        "macro_labels": len(label_results), "delta_auc": value,
                        "auc_cluster_se": interval["cluster_se"],
                        "auc_ci_lower": interval["ci_lower"], "auc_ci_upper": interval["ci_upper"]})
    return list(points.values()), effects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, help="Optional independently exported A directory")
    parser.add_argument("--feature-dir", type=Path,
                        default=ROOT / "artifacts/rise_sample_analysis/group_features/features")
    parser.add_argument("--frequency-dir", type=Path,
                        default=ROOT / "artifacts/rise_sample_analysis/group_features/frequencies")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimensions", nargs="+", default=["all"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    feature_paths = sorted(args.feature_dir.glob("part-*.parquet"))
    feature_columns = list(KEYS) + ["user_train_target_count", "item_train_target_count", "c1_entropy",
                                   "target_c1_history_fraction", "c1_ambiguous_count", "c1_known_count"]
    features = load_columns(feature_paths, feature_columns)
    feature_keys = unique_sorted_keys(features)
    inputs, receipts = {}, {}
    for variant in "BCDE":
        directory = args.prediction_dir / variant
        data, receipts[variant] = load_predictions(directory)
        if not np.array_equal(unique_sorted_keys(data), feature_keys):
            raise ValueError("Feature/prediction row identity mismatch")
        for key in KEYS:
            if not np.array_equal(data[key], features[key]):
                raise ValueError(f"Feature/prediction mismatch in {key}")
        inputs[variant] = data
    if args.baseline_dir:
        data, receipts["A"] = load_predictions(args.baseline_dir)
        if not np.array_equal(unique_sorted_keys(data), feature_keys):
            raise ValueError("Baseline row identity mismatch")
        for key in KEYS:
            if not np.array_equal(data[key], features[key]):
                raise ValueError(f"Baseline metadata mismatch: {key}")
        inputs["A"] = data
    for label in LABELS:
        if not all(np.array_equal(data[label], inputs["B"][label]) for data in inputs.values()):
            raise ValueError(f"Paired labels mismatch: {label}")
    points, effects, definitions = [], [], {}
    for dimension, membership, names, definition in dimensions(features, args.frequency_dir):
        if "all" not in args.dimensions and dimension not in args.dimensions:
            continue
        definitions[dimension] = {"groups": names, **definition,
                                  "included_rows": int(np.sum(membership >= 0))}
        for group in sorted(names):
            selected = membership == group
            n = int(selected.sum())
            if not n:
                definitions[dimension].setdefault("empty_groups", []).append(names[group])
                continue
            label_results = []
            for label in LABELS:
                y = inputs["B"][label][selected]
                prediction = {variant: data[label + "_pred"][selected] for variant, data in inputs.items()}
                result = summarize_group(y, prediction, features["user_id"][selected])
                label_results.append(result)
                pt, eff, _ = result
                tag = {"dimension": dimension, "group": names[group], "label": label}
                points += [{**tag, **row} for row in pt]
                effects += [{**tag, **row} for row in eff]
            pt, eff = macro_summary(label_results)
            tag = {"dimension": dimension, "group": names[group], "label": "macro_5_tasks"}
            points += [{**tag, **row} for row in pt]
            effects += [{**tag, **row} for row in eff]
            del label_results
            print(json.dumps({"dimension": dimension, "group": names[group], "rows": n}), flush=True)
        write_csv(args.output_dir / "group_metrics.csv", points)
        write_csv(args.output_dir / "group_effects.csv", effects)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "rows": len(feature_keys), "labels": LABELS, "variants": list(inputs),
                "definitions": definitions,
                "prediction_receipts": {variant: {key: value for key, value in receipt.items() if key != "sample_identity"}
                                        for variant, receipt in receipts.items()},
                "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                                  for path in (Path(__file__).resolve(), Path(__file__).with_name("statistical_metrics.py"))},
                "feature_files_sha256": {str(path): sha256(path) for path in feature_paths},
                "ci_method": "user-cluster linearized influence normal 95%; conditional on trained models",
                "statistical_claims": "Exploratory effect sizes; no independent training replication or multiplicity-corrected claims.",
                "action_support_analysis": "Deferred until attribution availability is resolved; no point-in-time claim."}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "complete.json").write_text(json.dumps({"rows": len(feature_keys),
                "metric_rows": len(points), "effect_rows": len(effects), "identity_verified": True}) + "\n")


if __name__ == "__main__":
    main()
