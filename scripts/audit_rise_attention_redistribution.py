"""Independently verify completed attention diagnostics and archive compact evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    complete = json.loads((source / "complete.json").read_text())
    if complete["completed"] is not True:
        raise ValueError("Export is incomplete")
    for name, digest in complete["files_sha256"].items():
        if sha256(source / name) != digest:
            raise ValueError(f"Artifact hash mismatch: {name}")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["inference_batch_size"] != 16384 or manifest["full_forward_rows_per_route"] != 131072:
        raise ValueError("Forward batches differ from the accepted recovery protocol")
    for name, digest in manifest["source_sha256"].items():
        if sha256(source / "frozen_sources" / name) != digest:
            raise ValueError(f"Frozen source mismatch: {name}")
    for name, digest in manifest["frozen_input_files"].items():
        if sha256(source / name) != digest:
            raise ValueError(f"Frozen input mismatch: {name}")
    accounting = subprocess.check_output(["sacct", "-j", args.job_id, "--format=JobID,State,ExitCode,NodeList,Elapsed,MaxRSS", "-P"], text=True)
    main_rows = [line.split("|") for line in accounting.splitlines()[1:] if line.split("|")[0] == args.job_id]
    if len(main_rows) != 1 or main_rows[0][1:3] != ["COMPLETED", "0:0"]:
        raise ValueError("Scheduler does not confirm successful completion")
    meta = pd.read_csv(source / "meta.csv")
    stats = pd.read_csv(source / "stats.csv")
    predictions = pd.read_csv(source / "prediction_check.csv")
    if len(meta) != 512 or len(stats) != 512 or len(predictions) != 512:
        raise ValueError("The frozen diagnostic requires 512 rows")
    for column in ("sample_index", "part_id", "row_offset", "user_id", "item_id", "effective_length", "is_click", "is_conversion"):
        if not np.array_equal(meta[column], predictions[column]):
            raise ValueError(f"Prediction identity mismatch: {column}")
    for part, group in meta.groupby("part_id"):
        ordered = sorted(range(16384), key=lambda row: hashlib.sha256(f"20260923:{part}:{row}".encode()).digest())
        if group.row_offset.tolist() != sorted(ordered[:64]):
            raise ValueError("Sampling does not match the frozen hash-priority rule")
    gates = {}
    for label in ("is_click", "is_conversion"):
        plain, captured, historical = [predictions[f"{label}_{route}"].to_numpy(dtype=np.float32)
                                      for route in ("plain", "captured", "historical")]
        if not np.isfinite(plain).all() or not np.array_equal(plain, captured):
            raise ValueError("Captured and plain predictions differ")
        if not np.allclose(plain, historical, atol=5e-4, rtol=5e-3):
            raise ValueError("Historical prediction gate failed")
        gates[label] = {"capture_bitwise_equal": True,
                        "historical_max_abs_error": float(np.max(np.abs(plain - historical)))}
    with np.load(source / "raw_attention.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    mask = data["valid_mask"][:, None, :]
    valid_rows = mask.any(-1)[:, 0]
    if not np.array_equal(mask.sum(-1)[:, 0], meta.effective_length):
        raise ValueError("Stored mask/length mismatch")
    if not np.array_equal(data["target_item_ids"], meta.item_id):
        raise ValueError("Target item identity mismatch")
    errors = {}
    for name, scores in (("weight_before", data["native_scores"]),
                         ("weight_after", data["native_scores"] + data["delta"])):
        logits = np.where(mask[valid_rows], scores[valid_rows], -np.inf).astype(np.float64)
        reconstructed = np.exp(logits - logits.max(-1, keepdims=True))
        reconstructed /= reconstructed.sum(-1, keepdims=True)
        actual = data[name][valid_rows]
        errors[name] = float(np.max(np.abs(reconstructed - actual)))
        if not np.allclose(reconstructed, actual, atol=2e-6, rtol=2e-5):
            raise ValueError("Independent float64 softmax reconstruction failed")
        if np.any(actual[~mask[valid_rows]] != 0):
            raise ValueError("Masked keys have nonzero mass")
        if not np.isnan(data[name][~valid_rows]).all():
            raise ValueError("Empty history has a fabricated distribution")
    tv = np.abs(data["weight_after"].astype(np.float64) - data["weight_before"]).sum(-1)[:, 0] / 2
    omega = np.max(np.where(mask, data["delta"], -np.inf), axis=-1)[:, 0] - np.min(np.where(mask, data["delta"], np.inf), axis=-1)[:, 0]
    if not np.allclose(tv, stats.tv, atol=2e-7, rtol=1e-6, equal_nan=True):
        raise ValueError("Stored TV statistics failed independent recomputation")
    if not np.allclose(omega[valid_rows], stats.omega[valid_rows], atol=1e-6, rtol=1e-6):
        raise ValueError("Stored correction ranges disagree")
    if np.any(tv[valid_rows] > np.tanh(omega[valid_rows] / 4) + 1e-6):
        raise ValueError("TV bound violated")
    top_before = data["weight_before"][valid_rows, 0].argmax(-1)
    top_after = data["weight_after"][valid_rows, 0].argmax(-1)
    if not np.array_equal(top_before, stats.top_before[valid_rows]) or not np.array_equal(top_after, stats.top_after[valid_rows]):
        raise ValueError("Stored top-key indices disagree")
    summary = {"examples": len(meta), "nonempty_examples": int(valid_rows.sum()),
               "empty_examples": int((~valid_rows).sum()), "unique_users": int(meta.user_id.nunique()),
               "tv_mean": float(np.mean(tv[valid_rows])), "tv_median": float(np.median(tv[valid_rows])),
               "tv_p10": float(np.quantile(tv[valid_rows], .1)), "tv_p90": float(np.quantile(tv[valid_rows], .9)),
               "top_key_changed_fraction": float(np.mean(top_before != top_after))}
    stored = json.loads((source / "summary.json").read_text())
    for key, value in summary.items():
        if not np.isclose(value, stored[key], atol=2e-7, rtol=1e-6):
            raise ValueError(f"Summary mismatch: {key}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("meta.csv", "stats.csv", "summary.json", "manifest.json", "prediction_gate.json", "historical_prediction_gate.json", "prediction_check.csv", "complete.json"):
        shutil.copy2(source / name, output / name)
    (output / "scheduler_accounting.txt").write_text(accounting)
    audit = {"passed": True, "job_id": args.job_id, "artifact_directory": str(source),
             "auditor_sha256": sha256(Path(__file__)), "source_hashes_verified": len(manifest["source_sha256"]),
             "artifact_hashes_verified": len(complete["files_sha256"]), "prediction_gates": gates,
             "independent_softmax_max_abs_error": errors, "recomputed_summary": summary}
    (output / "independent_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
