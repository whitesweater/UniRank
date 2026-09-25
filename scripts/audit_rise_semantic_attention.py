"""Independent numerical and source audit for the KuaiRand semantic diagnostic."""
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
    source, output = args.input_dir, args.output_dir
    complete = json.loads((source / "complete.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    assert complete["completed"] is True
    for name, expected in complete["files_sha256"].items():
        assert sha256(source / name) == expected, name
    for name, expected in manifest["source_sha256"].items():
        assert sha256(source / "frozen_sources" / name) == expected, name
    assert manifest["model"] == "RankMixer" and manifest["dataset"] == "KuaiRand_Video_Action"
    assert manifest["inference_batch_size"] == 16384 and manifest["examples"] == 512
    routes = manifest["route_info"]
    assert routes["plain"]["full_prediction_sha256"] == routes["capture"]["full_prediction_sha256"]
    assert all(routes[name]["full_forward_rows"] == 131072 for name in routes)
    assert routes["capture"]["selected_qkv_signatures_by_batch"] == routes["local_off"]["selected_qkv_signatures_by_batch"]
    assert routes["capture"]["site_info"] == routes["local_off"]["site_info"]
    assert routes["local_off"]["intervened_site"] == "attention_layers.dot_attention"
    accounting = subprocess.check_output(["sacct", "-j", args.job_id,
        "--format=JobID,State,ExitCode,Elapsed,NodeList", "-P"], text=True)
    row = [line.split("|") for line in accounting.splitlines() if line.startswith(args.job_id + "|")]
    assert len(row) == 1 and row[0][1:3] == ["COMPLETED", "0:0"]
    meta = pd.read_csv(source / "meta.csv", float_precision="round_trip")
    preds = pd.read_csv(source / "predictions.csv", float_precision="round_trip")
    sensitivity = pd.read_csv(source / "label_sensitivity.csv", float_precision="round_trip")
    assert len(meta) == len(preds) == 512 and not meta.duplicated(["part_id", "row_offset"]).any()
    for name in meta.columns:
        assert np.array_equal(meta[name], preds[name]), name
    for part, group in meta.groupby("part_id"):
        order = sorted(range(16384), key=lambda r: hashlib.sha256(f"20260923:{part}:{r}".encode()).digest())
        assert group.row_offset.tolist() == sorted(order[:64])
    loss_results = []
    for label in manifest["labels"]:
        on, captured, off = [preds[f"{label}_{route}"].to_numpy(dtype=np.float32)
                            for route in ("plain", "capture", "local_off")]
        assert np.array_equal(on, captured)
        y = meta[label].to_numpy()
        losses = []
        for name, p in (("plain", on), ("local_off", off)):
            assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
            p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
            loss = -y * np.log(p) - (1 - y) * np.log1p(-p)
            assert np.allclose(loss, preds[f"{label}_bce_{name}"], atol=1e-14, rtol=1e-13)
            losses.append(loss)
        gain = losses[1] - losses[0]
        assert np.allclose(gain, preds[f"{label}_bce_gain"], atol=1e-14, rtol=1e-13)
        recorded = sensitivity[(sensitivity.group == "all") & (sensitivity.label == label)].iloc[0]
        for name, value in (("improved", (gain > 0).sum()), ("degraded", (gain < 0).sum()),
                            ("unchanged", (gain == 0).sum()), ("positives", y.sum())):
            assert int(recorded[name]) == int(value)
        assert np.isclose(recorded.mean_bce_gain, gain.mean(), atol=1e-14, rtol=1e-13)
        loss_results.append({"label": label, "mean_bce_gain": float(gain.mean()),
                             "improved": int((gain > 0).sum()), "degraded": int((gain < 0).sum()),
                             "unchanged": int((gain == 0).sum()), "positives": int(y.sum())})
    with np.load(source / "raw_attention.npz", allow_pickle=False) as file:
        raw = dict(file)
    mask = raw["valid_mask"][:, None]
    assert np.array_equal(mask.sum(-1)[:, 0], meta.effective_length)
    assert np.array_equal(raw["target_item_ids"], meta.item_id)
    assert np.array_equal(raw["native_mask"], np.broadcast_to(mask, raw["native_mask"].shape))
    valid = mask.any(-1)[:, 0]
    errors = {}
    for name, scores in (("weight_before", raw["native_scores"]),
                         ("weight_after", raw["native_scores"] + raw["delta"])):
        logits = np.where(mask[valid], scores[valid], -np.inf).astype(np.float64)
        expected = np.exp(logits - logits.max(-1, keepdims=True))
        expected /= expected.sum(-1, keepdims=True)
        actual = raw[name][valid]
        errors[name] = float(np.max(np.abs(expected - actual)))
        assert np.allclose(expected, actual, atol=2e-6, rtol=2e-5)
        assert (actual[~np.broadcast_to(mask[valid], actual.shape)] == 0).all()
    tv = np.abs(raw["weight_after"].astype(np.float64) - raw["weight_before"]).sum(-1) / 2
    assert np.allclose(tv, raw["tv"], atol=2e-7, rtol=1e-6, equal_nan=True)
    omega = np.where(mask, raw["delta"], -np.inf).max(-1) - np.where(mask, raw["delta"], np.inf).min(-1)
    assert np.all(tv[valid] <= np.tanh(omega[valid] / 4) + 1e-6)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("meta.csv", "predictions.csv", "stats.csv", "label_sensitivity.csv", "manifest.json",
                 "summary.json", "prediction_gate.json", "local_intervention_gate.json", "complete.json"):
        shutil.copy2(source / name, output / name)
    (output / "scheduler_accounting.txt").write_text(accounting)
    result = {"passed": True, "job_id": args.job_id, "examples": 512, "unique_users": int(meta.user_id.nunique()),
              "prediction_bitwise_equal": True, "full_prediction_hashes_equal": True,
              "local_off_qkv_hashes_equal": True, "softmax_max_abs_error": errors,
              "independent_label_sensitivity": loss_results, "auditor_sha256": sha256(Path(__file__)),
              "source_hashes_verified": len(manifest["source_sha256"]),
              "artifact_hashes_verified": len(complete["files_sha256"])}
    (output / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
