"""Analyze fixed-checkpoint history-order probes without training or GPU use.

This script reads raw artifacts and writes a new analysis directory. It never
changes experiment inputs, checkpoints, sources, or existing result files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from permutation_statistics import (bce, clusters, cluster_sums, bootstrap_weights,
                                    paired_cluster_summary, diagnostic_auc)

CASES = {
    "rankmixer_kuairand": ["is_click", "is_follow", "is_like", "is_comment", "is_forward", "long_view"],
    "hiformer_taac25": ["is_click", "is_conversion"],
}
ROUTES = ("identity", "off", "reverse", "random")
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_DRAWS = 2000


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def strict_bool(series):
    if series.dtype == bool:
        return series.to_numpy()
    mapping = {"True": True, "False": False, "true": True, "false": False}
    if not series.astype(str).isin(mapping).all():
        raise ValueError(f"Not a strict Boolean field: {series.name}")
    return series.astype(str).map(mapping).to_numpy(dtype=bool)


def equal_with_missing(left, right):
    """CSV missing values compare equal without conflating different row values."""
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return False
    missing_left, missing_right = pd.isna(left), pd.isna(right)
    return bool(np.array_equal(missing_left, missing_right) and np.array_equal(left[~missing_left], right[~missing_right]))


def check_completion(source, required_files, errors, receipts, historical=False):
    path = source / "complete.json"
    complete = read_json(path)
    if complete.get("completed") is not True or (historical and complete.get("all_gates_passed") is not True) or complete.get("all_gates_passed", True) is not True:
        errors.append({"check": "completion", "path": str(path), "detail": "Completed receipt (and historical all-gates pass) required"})
    if not historical and complete.get("no_training") is not True:
        errors.append({"check": "no_training_receipt", "path": str(path)})
    hashes = complete.get("files_sha256", {})
    for filename in required_files:
        if filename not in hashes:
            errors.append({"check": "missing_hash", "path": str(source / filename)})
    for filename, expected in hashes.items():
        target = (source / filename).resolve()
        if not target.is_relative_to(source.resolve()):
            raise ValueError("Artifact hash path escapes result directory")
        if not target.is_file():
            errors.append({"check": "missing_hashed_artifact", "path": str(target)})
        else:
            actual = sha(target)
            receipts[str(target)] = actual
            if actual != expected:
                errors.append({"check": "artifact_hash", "path": str(target), "expected": expected, "actual": actual})
    receipts[str(path)] = sha(path)
    return complete


def verify_runtime_gates(gates, labels, errors):
    bitwise_prediction_gates = ("plain_vs_historical_attention512", "identity_vs_plain_allrows",
                               "identity_vs_prior_probe_full_allrows", "off_vs_prior_probe_local_off_allrows")
    for key in bitwise_prediction_gates:
        values = gates.get(key, {})
        if set(values) != set(labels) or not all(v.get("equal") is True and v.get("max_abs_error") == 0 for v in values.values()):
            errors.append({"check": key, "detail": values})
    if gates.get("historical_attention512_raw_bitwise") is not True:
        errors.append({"check": "historical_attention512_raw_bitwise"})
    for route in ("off", "reverse", "random"):
        record = gates.get(route, {})
        if record.get("native_qkv_mask_scale_bitwise") is not True or record.get("native_scores_bitwise") is not True:
            errors.append({"check": "native_path_fixed", "route": route, "detail": record})
    floor = gates.get("native_joint_permutation_numerical_floor", {})
    if floor.get("passed_fp32_pooling_gate") is not True or floor.get("rows") != 64:
        errors.append({"check": "native_joint_permutation_fp32", "detail": floor})
    required = {f"{precision}_{route}" for precision in ("fp32_math", "bf16_eager_auto_sdpa") for route in ("identity", "reverse", "random")}
    if set(floor.get("checks", {})) != required:
        errors.append({"check": "native_joint_permutation_coverage"})
    for key, record in floor.get("checks", {}).items():
        if set(record.get("prediction", {})) != set(labels):
            errors.append({"check": "native_joint_prediction_labels", "mode": key})
        for name, summary in [("pooling", record["pooling"])] + list(record["prediction"].items()):
            if not all(isinstance(summary.get(metric), (int, float)) and np.isfinite(summary[metric]) and summary[metric] >= 0 for metric in ("max_abs", "mean_abs")):
                errors.append({"check": "native_numerical_floor_finite", "mode": key, "quantity": name})
            if key.endswith("_identity") and (summary.get("max_abs") != 0 or summary.get("bitwise_equal") is not True):
                errors.append({"check": "native_identity_self_equality", "mode": key, "quantity": name})


def verify_permutation_files(source, plan, errors):
    with np.load(source / "permutations.npz", allow_pickle=False) as files:
        if set(files.files) != {"reverse", "random"}:
            errors.append({"check": "permutation_route_keys"})
        for route in ("reverse", "random"):
            value = files[route]
            if value.shape != (131072, 100):
                errors.append({"check": "permutation_shape", "route": route, "shape": value.shape})
                continue
            original = np.broadcast_to(np.arange(100), value.shape)
            if not np.array_equal(np.sort(value, axis=1), original):
                errors.append({"check": "permutation_bijection", "route": route})
            hashes = [hashlib.sha256(np.ascontiguousarray(chunk).tobytes()).hexdigest() for chunk in np.split(value, 8)]
            if hashes != plan.get("permutation_hashes", {}).get(route):
                errors.append({"check": "frozen_permutation_hashes", "route": route})


def analyze_population(frame, labels, mask, population):
    subset = frame.loc[mask]
    ids, inverse, counts = clusters(subset.user_id.to_numpy())
    draws = bootstrap_weights(len(ids), BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
    targets = subset[labels].to_numpy()
    losses, sums, predictions = {}, {}, {}
    metric_rows, contrasts, change_rows, cluster_rows = [], [], [], []
    for route in ROUTES:
        p = subset[[f"{label}_{route}" for label in labels]].to_numpy(dtype=np.float64)
        values = bce(targets, p)
        values = np.column_stack([values, values.mean(axis=1)])
        losses[route], predictions[route] = values, p
        sums[route] = cluster_sums(values, inverse, len(ids))
        for index, label in enumerate(labels + ["label_macro"]):
            metric_rows.append({"population": population, "route": route, "label": label,
                "rows": len(subset), "users": len(ids),
                "positive_rows": int(targets[:, index].sum()) if index < len(labels) else None,
                "mean_bce": float(values[:, index].mean()),
                "user_balanced_bce": float((sums[route][:, index] / counts).mean()),
                "diagnostic_auc": diagnostic_auc(targets[:, index], p[:, index]) if index < len(labels) else None})
            cluster_rows.append(pd.DataFrame({"population": population, "user_id": ids,
                "rows": counts.astype(int), "route": route, "label": label,
                "loss_sum": sums[route][:, index]}))
    for route in ("reverse", "random", "off"):
        summaries = paired_cluster_summary(sums[route] - sums["identity"], counts, draws)
        abs_change = np.abs(predictions[route] - predictions["identity"])
        abs_change = np.column_stack([abs_change, abs_change.mean(axis=1)])
        abs_sums = cluster_sums(abs_change, inverse, len(ids))
        for index, label in enumerate(labels + ["label_macro"]):
            contrasts.append({"population": population, "contrast": f"{route}_minus_identity", "label": label,
                "rows": len(subset), "users": len(ids), **summaries[index],
                "primary": population == "history_length_ge_2" and label == "is_click" and route in ("reverse", "random"),
                "supporting_utility": population == "history_length_ge_2" and label == "is_click" and route == "off"})
            change = abs_change[:, index]
            change_rows.append({"population": population, "route": route, "label": label,
                "mean_abs_prediction_change": float(change.mean()),
                "user_balanced_mean_abs_prediction_change": float((abs_sums[:, index] / counts).mean()),
                "median_abs_prediction_change": float(np.median(change)),
                "p90_abs_prediction_change": float(np.quantile(change, .9)),
                "max_abs_prediction_change": float(change.max()),
                "fraction_predictions_changed_exactly": float(np.mean(change > 0))})
    user_summary = {"rows": len(subset), "unique_users": len(ids), "min_rows_per_user": int(counts.min()),
        "median_rows_per_user": float(np.median(counts)), "max_rows_per_user": int(counts.max()),
        "largest_user_fraction": float(counts.max() / counts.sum()),
        "effective_users_by_row_concentration": float(counts.sum() ** 2 / (counts @ counts))}
    user_frame = pd.DataFrame({"population": population, "user_id": ids, "rows": counts.astype(int)})
    return metric_rows, contrasts, change_rows, cluster_rows, user_frame, user_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--historical-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--historical-prediction-atol", type=float, default=1e-5,
                        help="Predeclared maximum absolute fidelity tolerance; default 1e-5, not fitted to outcomes")
    args = parser.parse_args()
    source, historical = args.results.resolve(), args.historical_results.resolve()
    labels = CASES[args.case]
    errors, receipts, gate_details = [], {}, {}
    complete = check_completion(source, ("predictions.csv", "meta.csv", "plan.json", "gates.json", "manifest.json", "permutations.npz"), errors, receipts)
    old_complete = check_completion(historical, ("predictions.csv", "meta.csv", "plan.json"), errors, receipts, historical=True)
    plan = read_json(source / "plan.json")
    gates = read_json(source / "gates.json")
    verify_runtime_gates(gates, labels, errors)
    verify_permutation_files(source, plan, errors)
    if plan.get("no_training") is not True or plan.get("case") != args.case or plan.get("labels") != labels or plan.get("rows") != 131072:
        errors.append({"check": "frozen_protocol_identity"})
    if Path(plan.get("prior_probe_results", "")).resolve() != historical:
        errors.append({"check": "historical_results_path", "declared": plan.get("prior_probe_results"), "actual": str(historical)})
    watched = plan.get("frozen_files_sha256", {})
    if not watched:
        errors.append({"check": "missing_frozen_sources"})
    for filename, expected in watched.items():
        path = Path(filename)
        if not path.is_file():
            errors.append({"check": "missing_watched_source", "path": filename})
            continue
        actual = sha(path)
        receipts[filename] = actual
        if actual != expected:
            errors.append({"check": "watched_source_hash", "path": filename, "expected": expected, "actual": actual})
    if sha(source / "meta.csv") != plan.get("meta_csv_sha256"):
        errors.append({"check": "frozen_metadata_hash"})
    if args.expected_plan_sha256 and sha(source / "plan.json") != args.expected_plan_sha256:
        errors.append({"check": "frozen_plan_identity", "expected": args.expected_plan_sha256, "actual": sha(source / "plan.json")})
    frame = pd.read_csv(source / "predictions.csv", float_precision="round_trip")
    meta = pd.read_csv(source / "meta.csv", float_precision="round_trip")
    old_frame = pd.read_csv(historical / "predictions.csv", float_precision="round_trip")
    old_meta = pd.read_csv(historical / "meta.csv", float_precision="round_trip")
    if len(frame) != 131072 or len(meta) != len(frame) or len(old_meta) != len(frame) or len(old_frame) != len(frame):
        errors.append({"check": "fixed_row_counts", "counts": [len(frame), len(meta), len(old_frame), len(old_meta)]})
    if not np.array_equal(frame.row_index, np.arange(len(frame))):
        errors.append({"check": "contiguous_row_ids"})
    for column in meta.columns:
        if column not in frame or not equal_with_missing(frame[column], meta[column]):
            errors.append({"check": "prediction_metadata_identity", "column": column})
    identity_columns = ["row_index", "part_id", "row_offset", "user_id", "item_id", "seq_len", "effective_length"] + labels
    for column in identity_columns:
        if column not in meta or column not in old_meta:
            errors.append({"check": "missing_identity_field", "column": column})
        elif not equal_with_missing(meta[column], old_meta[column]) or not equal_with_missing(old_meta[column], old_frame[column]):
            errors.append({"check": "historical_input_identity", "column": column})
    effective = frame.effective_length.to_numpy()
    if not np.isfinite(effective).all() or not np.equal(effective, np.floor(effective)).all() or (effective < 0).any():
        errors.append({"check": "history_length_validity"})
    eligible = effective >= 2
    if complete.get("rows") != len(frame) or complete.get("primary_eligible_rows") != int(eligible.sum()) or plan.get("primary_eligible_rows") != int(eligible.sum()):
        errors.append({"check": "declared_population_counts"})
    if not np.array_equal(strict_bool(frame.history_permutation_eligible), eligible) or not np.array_equal(strict_bool(meta.history_permutation_eligible), eligible):
        errors.append({"check": "input_only_eligibility"})
    prior_eligible = strict_bool(old_meta.eligible) if "eligible" in old_meta else np.ones(len(old_meta), dtype=bool)
    eligibility_difference = frame.loc[eligible != prior_eligible, identity_columns].copy()
    eligibility_difference["history_length_ge_2"] = eligible[eligible != prior_eligible]
    eligibility_difference["prior_peer_eligible"] = prior_eligible[eligible != prior_eligible]
    gate_details["eligibility"] = {"new_eligible_rows": int(eligible.sum()), "prior_eligible_rows": int(prior_eligible.sum()),
        "different_row_count": int((eligible != prior_eligible).sum()),
        "same_row_ids": bool(np.array_equal(frame.loc[eligible, "row_index"], old_meta.loc[prior_eligible, "row_index"]))}
    historical_comparison = []
    for route, old_route in (("identity", "full"), ("off", "local_off")):
        for label in labels:
            actual = frame[f"{label}_{route}"].to_numpy(dtype=np.float64)
            original = old_frame[f"{label}_{old_route}"].to_numpy(dtype=np.float64)
            delta = np.abs(actual - original)
            row = {"route": route, "historical_route": old_route, "label": label,
                "rows": len(delta), "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                "bitwise_equal": bool(np.array_equal(actual, original)), "declared_atol": args.historical_prediction_atol,
                "float32_exact_equal": bool(np.array_equal(actual.astype(np.float32), original.astype(np.float32))),
                "passed": bool((delta <= args.historical_prediction_atol).all() and np.array_equal(actual.astype(np.float32), original.astype(np.float32)))}
            historical_comparison.append(row)
            if not row["passed"]:
                errors.append({"check": "historical_prediction_fidelity", **row})
    for route in ROUTES:
        bce(frame[labels].to_numpy(), frame[[f"{label}_{route}" for label in labels]].to_numpy())
    for route in ("reverse", "random"):
        ineligible_change = np.abs(frame.loc[~eligible, [f"{label}_{route}" for label in labels]].to_numpy() - frame.loc[~eligible, [f"{label}_identity" for label in labels]].to_numpy())
        gate_details[f"{route}_ineligible"] = {"rows": int((~eligible).sum()), "max_abs": float(ineligible_change.max()) if ineligible_change.size else 0.}
        if np.any(ineligible_change != 0):
            errors.append({"check": "short_history_no_permutation_effect", "route": route})
    gate_details["runtime_gates"] = gates
    gate_details["note"] = "Runtime gate fields and values checked; independent source review validates their implementations. BF16 numerical residuals use eager modules with automatic SDPA, not the compiled production path; no bitwise invariance assertion."
    args.output.mkdir(parents=True, exist_ok=False)
    audit = {"status": "FAILED" if errors else "RAW_DATA_VERIFIED", "case": args.case,
        "errors": errors, "gate_details": gate_details, "historical_prediction_comparison": historical_comparison,
        "source_hashes_recomputed": receipts, "script_sha256": sha(__file__),
        "statistics_script_sha256": sha(Path(__file__).with_name("permutation_statistics.py")),
        "analysis_plan_sha256": sha(Path(__file__).with_name("analysis_plan.json")),
        "expected_plan_sha256_supplied": args.expected_plan_sha256,
        "interpretation_requires_runtime_gate_review": True}
    (args.output / "independent_data_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    eligibility_difference.to_csv(args.output / "eligibility_differences.csv", index=False)
    if errors:
        print(json.dumps({"status": "FAILED", "errors": len(errors), "output": str(args.output)}))
        return 2
    metrics, contrasts, changes, cluster_frames, user_frames, population_summaries = [], [], [], [], [], {}
    for population, mask in (("history_length_ge_2", eligible), ("all_rows", np.ones(len(frame), dtype=bool))):
        mm, cc, pp, cl, users, summary = analyze_population(frame, labels, mask, population)
        metrics.extend(mm); contrasts.extend(cc); changes.extend(pp); cluster_frames.extend(cl); user_frames.append(users)
        population_summaries[population] = summary
    pd.DataFrame(metrics).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(contrasts).to_csv(args.output / "contrasts.csv", index=False)
    pd.DataFrame(changes).to_csv(args.output / "prediction_changes.csv", index=False)
    pd.concat(cluster_frames, ignore_index=True).to_csv(args.output / "cluster_loss_sums.csv", index=False)
    pd.concat(user_frames, ignore_index=True).to_csv(args.output / "user_clusters.csv", index=False)
    record = {"status": "complete", "case": args.case, "rows": len(frame), "populations": population_summaries,
        "model": plan.get("model"), "dataset": plan.get("dataset"), "checkpoint_sha256": plan.get("checkpoint_sha256"),
        "intervention_scope": plan.get("route_definition"), "off_scope": plan.get("off_definition"),
        "history_order_scope": plan.get("native_order_wording", "data-provided/native order; chronology is not presumed"),
        "native_numerical_check_scope": "64-row eager modules with FP32 MATH and BF16 automatic SDPA; the BF16 check is not a compiled production-path test",
        "primary_endpoints": [v for v in contrasts if v["primary"]],
        "supporting_utility_endpoint": next(v for v in contrasts if v["supporting_utility"]),
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_interval": "95 percent percentile user-cluster interval",
        "uncertainty": "Conditional on one trained checkpoint, the fixed sample window, and one fixed reverse/random intervention; no training, permutation-distribution, or population uncertainty claim.",
        "direction": "BCE(route) minus BCE(identity): positive means this route has worse loss. Mean absolute prediction differences measure sensitivity, not benefit.",
        "interpretation": "Correction-history order sensitivity and on/off predictive utility are separate findings. These interventions do not prove order modeling accounts for all gains.",
        "historical_eligibility_comparison": gate_details["eligibility"],
        "input_predictions_sha256": sha(source / "predictions.csv"), "plan_sha256": sha(source / "plan.json"),
        "analysis_audit_sha256": sha(args.output / "independent_data_audit.json")}
    (args.output / "summary.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    files = {str(p.name): sha(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / "analysis_manifest.json").write_text(json.dumps({"files_sha256": files}, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
