"""Synthetic file/schema test only; generated data are never scientific evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from permutation_statistics import bce


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def complete(directory, **fields):
    files = {str(p.relative_to(directory)): sha(p) for p in directory.rglob("*") if p.is_file() and p != directory / "complete.json"}
    write_json(directory / "complete.json", {"completed": True, "all_gates_passed": True, "files_sha256": files, **fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=False)
    prior, source = args.workspace / "synthetic_prior", args.workspace / "synthetic_probe"
    prior.mkdir(); source.mkdir()
    n = 131072
    labels = ["is_click", "is_conversion"]
    length = np.arange(n) % 4
    eligible = length >= 2
    prior_eligible = eligible.copy()
    prior_eligible[2] = False
    meta = pd.DataFrame({"row_index": np.arange(n), "part_id": np.arange(n) // 16384,
        "row_offset": np.arange(n) % 16384, "user_id": np.arange(n) % 4,
        "item_id": np.arange(n) + 10, "seq_len": length, "effective_length": length,
        "eligible": prior_eligible, "exclusion_reason": [None] * n,
        "is_click": np.arange(n) % 2, "is_conversion": (np.arange(n) + 1) % 2})
    old_frame = meta.copy()
    for label in labels:
        old_frame[label + "_full"] = np.float32(.4)
        old_frame[label + "_local_off"] = np.float32(.5)
    meta.to_csv(prior / "meta.csv", index=False)
    old_frame.to_csv(prior / "predictions.csv", index=False)
    write_json(prior / "plan.json", {"synthetic_test_only": True})
    original_score = np.full((n, 100), .125, dtype=np.float32)
    np.savez_compressed(prior / "attention_full.npz", delta=original_score[:, None, :])
    complete(prior, rows=n)
    meta["history_permutation_eligible"] = eligible
    frame = meta.copy()
    for label in labels:
        for route, value in (("identity", .4), ("off", .5), ("reverse", .45), ("random", .35)):
            frame[label + "_" + route] = np.where(eligible if route in ("reverse", "random") else True, np.float32(value), np.float32(.4)).astype(np.float32)
    meta.to_csv(source / "meta.csv", index=False)
    frame.to_csv(source / "predictions.csv", index=False)
    identity = np.broadcast_to(np.arange(100, dtype=np.int16), (n, 100)).copy()
    reverse, random = identity.copy(), identity.copy()
    reverse[eligible, :2] = np.array([1, 0], dtype=np.int16)
    random[eligible, :2] = np.array([1, 0], dtype=np.int16)
    np.savez_compressed(source / "permutations.npz", reverse=reverse, random=random)
    permutation_hashes = {route: [hashlib.sha256(np.ascontiguousarray(chunk).tobytes()).hexdigest() for chunk in np.split(value, 8)]
                          for route, value in (("reverse", reverse), ("random", random))}
    revision = "native_capture_original_input_replay_v2"
    score_arrays = {"identity": original_score, "off": np.zeros_like(original_score),
                    "reverse": np.full_like(original_score, .15), "random": np.full_like(original_score, .1)}
    np.savez_compressed(source / "score_bias_by_route.npz", **score_arrays)
    first = source / "first_window_fidelity"
    first.mkdir()
    fidelity_plan_path = args.workspace / "plan_fidelity_v2_hiformer_taac25.json"
    checkpoint_sha = "a" * 64
    write_json(fidelity_plan_path, {"fidelity_only": True, "full_history_study": False, "checkpoint_sha256": checkpoint_sha})
    shutil.copyfile(fidelity_plan_path, first / "plan.json")
    plan = {"schema_version": 2, "protocol_revision": revision, "no_training": True, "case": "hiformer_taac25", "labels": labels, "rows": n,
            "primary_eligible_rows": int(eligible.sum()), "prior_probe_results": str(prior.resolve()),
            "meta_csv_sha256": sha(source / "meta.csv"), "permutation_hashes": permutation_hashes,
            "checkpoint_sha256": checkpoint_sha,
            "frozen_files_sha256": {str((prior / "predictions.csv").resolve()): sha(prior / "predictions.csv"),
                                    str(fidelity_plan_path.resolve()): sha(fidelity_plan_path)},
            "synthetic_test_only": True}
    write_json(source / "plan.json", plan)
    gates = {key: {label: {"equal": True, "max_abs_error": 0.} for label in labels} for key in
        ("plain_vs_historical_attention512", "identity_vs_plain_allrows", "identity_vs_prior_probe_full_allrows", "off_vs_prior_probe_local_off_allrows", "native_capture_vs_fresh_plain_allrows", "native_capture_vs_prior_probe_full_allrows")}
    gates["historical_attention512_raw_bitwise"] = True
    gates["first_window_fidelity_passed"] = True
    gates["identity_replay_actual_bias_exact"] = True
    for route in ("off", "reverse", "random"):
        gates[route] = {"native_qkv_mask_scale_bitwise": True, "native_scores_bitwise": True, "actual_bias_equals_intended": True}
        if route != "off":
            gates[route]["captured_native_content_inverse_alignment"] = True
    zero = {"max_abs": 0., "mean_abs": 0., "bitwise_equal": True}
    gates["native_joint_permutation_numerical_floor"] = {"passed_fp32_pooling_gate": True, "rows": 64,
        "checks": {f"{precision}_{route}": {"pooling": zero, "prediction": {label: zero for label in labels}}
            for precision in ("fp32_math", "bf16_eager_auto_sdpa") for route in ("identity", "reverse", "random")}}
    write_json(source / "gates.json", gates)
    native = {"qkv": {key: {"shape": [16384, 1, 100, 16], "dtype": "torch.bfloat16", "sha256": "1" * 64} for key in ("query", "key", "value")},
              "site": {"scale": .25, "mask": {"sha256": "2" * 64}, "bias_dtype": "torch.bfloat16", "mask_dtype": "torch.bool"}}
    first_meta = old_frame.iloc[:16384][[c for c in old_frame if not c.endswith(("_full", "_local_off"))]]
    first_meta.to_csv(first / "meta.csv", index=False)
    first_predictions = {f"{label}_{route}": old_frame.iloc[:16384][f"{label}_{'local_off' if route == 'off' else 'full'}"].to_numpy(dtype=np.float32)
                         for label in labels for route in ("plain", "capture", "replay", "off")}
    np.savez_compressed(first / "fidelity_predictions.npz", **first_predictions)
    first_gates = {key: {label: {"equal": True, "max_abs_error": 0.} for label in labels} for key in
                   ("plain_vs_prior_full", "capture_vs_plain", "replay_vs_plain", "zero_vs_prior_local_off")}
    first_gates.update({key: True for key in ("replay_bias_exact", "replay_qkv_mask_scale_exact", "zero_qkv_mask_scale_exact", "zero_bias_exact")})
    write_json(first / "gates.json", first_gates)
    write_json(first / "manifest.json", {"fidelity_only": True, "no_permutation_outcome_inspected": True, "no_training": True,
        "plan_sha256": sha(fidelity_plan_path), "capture_info": [native], "replay_info": [native], "zero_info": [native]})
    complete(first, fidelity_only=True, rows=16384, no_training=True)
    alignment = {route: [{"original_order_qkv_recovered_bitwise": True, "aligned_qkv": native["qkv"], "captured_native": native,
                         "aligned_delta_sha256": hashlib.sha256(np.ascontiguousarray(chunk).tobytes()).hexdigest()} for chunk in np.split(score_arrays[route], 8)]
                 for route in ("reverse", "random")}
    write_json(source / "manifest.json", {"synthetic_test_only": True, "no_training": True, "protocol_revision": revision,
        "plan_sha256": sha(source / "plan.json"), "route_info": {route: [native] * 8 for route in ("identity", "off", "reverse", "random")},
        "capture_alignment": alignment})
    complete(source, rows=n, primary_eligible_rows=int(eligible.sum()), no_training=True, protocol_revision=revision)
    analysis = args.workspace / "synthetic_analysis"
    command = [sys.executable, str(Path(__file__).with_name("analyze_permutation.py")), str(source),
        "--case", "hiformer_taac25", "--historical-results", str(prior),
        "--expected-plan-sha256", sha(source / "plan.json"), "--output", str(analysis)]
    run = subprocess.run(command, capture_output=True, text=True)
    (args.workspace / "synthetic_success_run.log").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise AssertionError(f"Synthetic valid fixture failed: {run.stdout[-4000:]} {run.stderr[-4000:]}")
    summary = json.loads((analysis / "summary.json").read_text())
    assert summary["historical_eligibility_comparison"]["different_row_count"] == 1
    assert len(summary["primary_endpoints"]) == 2
    for result in summary["primary_endpoints"]:
        route = result["contrast"].split("_minus_")[0]
        direct = (bce(frame.loc[eligible, "is_click"].to_numpy(), frame.loc[eligible, "is_click_" + route].to_numpy())
                  - bce(frame.loc[eligible, "is_click"].to_numpy(), frame.loc[eligible, "is_click_identity"].to_numpy())).mean()
        np.testing.assert_allclose(result["bce_difference"], direct, atol=1e-7, rtol=0)
    # Rehash corrupted data to ensure the metadata identity gate, not only SHA,
    # catches it. This mutation is confined to synthetic test input.
    frame.loc[0, "row_index"] = -1
    frame.to_csv(source / "predictions.csv", index=False)
    complete(source, rows=n, primary_eligible_rows=int(eligible.sum()), no_training=True, protocol_revision=revision)
    command[-1] = str(args.workspace / "synthetic_rejected_analysis")
    invalid = subprocess.run(command, capture_output=True, text=True)
    (args.workspace / "synthetic_rejection_run.log").write_text(invalid.stdout + invalid.stderr)
    assert invalid.returncode == 2
    rejection = json.loads((args.workspace / "synthetic_rejected_analysis/independent_data_audit.json").read_text())
    assert any(v["check"] == "contiguous_row_ids" for v in rejection["errors"])
    # A claimed passed gate and updated file hashes must not conceal changed
    # first-window raw predictions. Restore row identity before this check.
    frame.loc[0, "row_index"] = 0
    frame.to_csv(source / "predictions.csv", index=False)
    first_predictions["is_click_replay"] = first_predictions["is_click_replay"].copy()
    first_predictions["is_click_replay"][0] += np.float32(.1)
    np.savez_compressed(first / "fidelity_predictions.npz", **first_predictions)
    complete(first, fidelity_only=True, rows=16384, no_training=True)
    complete(source, rows=n, primary_eligible_rows=int(eligible.sum()), no_training=True, protocol_revision=revision)
    command[-1] = str(args.workspace / "synthetic_rejected_firstwindow_analysis")
    bad_first = subprocess.run(command, capture_output=True, text=True)
    (args.workspace / "synthetic_firstwindow_rejection_run.log").write_text(bad_first.stdout + bad_first.stderr)
    assert bad_first.returncode == 2
    first_rejection = json.loads((args.workspace / "synthetic_rejected_firstwindow_analysis/independent_data_audit.json").read_text())
    assert any(v["check"] == "v2_first_window_prediction_exact" for v in first_rejection["errors"])
    write_json(args.workspace / "synthetic_test_receipt.json", {"synthetic_test_only": True,
        "passed": True, "valid_fixture_exit": run.returncode, "corrupt_fixture_exit": invalid.returncode,
        "corrupt_firstwindow_exit": bad_first.returncode,
        "checks": ["CSV NaNs", "v2 nested first-window raw NPZ fidelity", "inverse-aligned score hashes and native QKV receipts", "runtime gate schema", "changed eligibility row IDs", "direct paired BCE formula", "rehash does not conceal row corruption", "rehash and passed gate do not conceal raw first-window prediction corruption"]})
    print("Synthetic file/schema test passed; these inputs are not experiment evidence.")


if __name__ == "__main__":
    main()
