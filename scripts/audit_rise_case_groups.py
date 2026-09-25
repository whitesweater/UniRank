"""Audit actual attention-mass interventions and make bounded case decisions."""
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    source, out = args.input_dir, args.output_dir
    complete = json.loads((source / "complete.json").read_text())
    assert complete["completed"] and complete["all_gates_passed"]
    for name, digest in complete["files_sha256"].items():
        assert sha256(source / name) == digest, name
    manifest = json.loads((source / "manifest.json").read_text())
    groups = json.loads((source / "groups.json").read_text())
    if "active_protocol" in groups:
        assert sha256(Path(groups["active_protocol"])) == groups["active_protocol_sha256"]
    gates = json.loads((source / "gates.json").read_text())
    assert all(value["passed"] for value in gates.values())
    assert gates["full_hook"]["all_window_predictions_equal"]
    assert gates["full_hook"]["qkv_equal_source"] and gates["full_hook"]["raw_equal_source"]
    for route, gate in gates.items():
        if "nonprobe_predictions" in gate:
            assert gate["all_selected_qkv_equal_full"]
            assert all(x["bitwise_equal"] for x in gate["nonprobe_predictions"])
    accounting = subprocess.check_output(["sacct", "-j", args.job_id,
        "--format=JobID,State,ExitCode,Elapsed,NodeList", "-P"], text=True)
    row = [r.split("|") for r in accounting.splitlines() if r.startswith(args.job_id + "|")]
    assert len(row) == 1 and row[0][1:3] == ["COMPLETED", "0:0"]
    predictions = pd.read_csv(source / "predictions.csv", float_precision="round_trip")
    masses = pd.read_csv(source / "group_mass.csv", float_precision="round_trip")
    raw_by_route = {}
    reconstruction_errors = {}
    for route in predictions.route.unique():
        with np.load(source / f"attention_{route}.npz", allow_pickle=False) as file:
            raw = dict(file)
        raw_by_route[route] = raw
        mask = raw["native_mask"]
        scores = raw["native_scores"] + raw["delta"]
        logits = np.where(mask, scores, -np.inf).astype(np.float64)
        expected = np.exp(logits - logits.max(-1, keepdims=True))
        expected /= expected.sum(-1, keepdims=True)
        reconstruction_errors[route] = float(np.max(np.abs(expected - raw["weight_after"])))
        assert np.allclose(expected, raw["weight_after"], atol=2e-6, rtol=2e-5)
        assert np.array_equal(raw["native_scores"], raw_by_route["full_hook"]["native_scores"])
        assert np.array_equal(raw["weight_before"], raw_by_route["full_hook"]["weight_before"])
        assert np.array_equal(raw["sample_indices"], [c["sample_index"] for c in groups["cases"]])
        for slot, case in enumerate(groups["cases"]):
            i = case["sample_index"]
            g, c = case["related_positions"], case["control_positions"]
            r = np.flatnonzero(mask[slot, 0] & ~np.isin(np.arange(100), g + c))
            recorded = masses[(masses.sample_index == i) & (masses.route == route)].iloc[0]
            for name, positions in (("related_mass", g), ("control_mass", c), ("receiver_mass", r)):
                assert np.isclose(expected[slot, 0, positions].sum(), recorded[name], atol=2e-6, rtol=0)
    loss = -(predictions.truth.to_numpy() * np.log(np.clip(predictions.prediction.to_numpy(), 1e-7, 1 - 1e-7))
             + (1 - predictions.truth.to_numpy()) * np.log1p(-np.clip(predictions.prediction.to_numpy(), 1e-7, 1 - 1e-7)))
    assert np.allclose(loss, predictions.bce, atol=1e-13, rtol=1e-13)
    source_dir = Path(manifest["source_dir"])
    old = pd.read_csv(source_dir / "predictions.csv", float_precision="round_trip")
    for _, row in predictions.iterrows():
        reference = old.iloc[int(row.sample_index)]
        assert row.truth == reference[row.label]
        assert np.float32(row.source_local_off_prediction) == np.float32(reference[f"{row.label}_local_off"])
        assert np.isclose(row.source_local_off_bce, reference[f"{row.label}_bce_local_off"], atol=1e-13)
        if row.route == "full_hook":
            assert np.float32(row.prediction) == np.float32(reference[f"{row.label}_plain"])
    decisions = []
    for slot, case in enumerate(groups["cases"]):
        i = case["sample_index"]
        p = predictions[(predictions.sample_index == i) & (predictions.label == "is_click")].set_index("route")
        m = masses[masses.sample_index == i].set_index("route")
        full = p.loc["full_hook"]
        gain = full.source_local_off_bce - full.bce
        assert gain > 0 and full.truth == 1
        ratio = (p.loc["related_revert"].bce - full.bce) / gain
        g_equal_loss = p.loc["related_equal_mass"].bce
        c_equal_loss = p.loc["control_equal_mass"].bce
        mg = m.loc["full_hook"].related_mass - m.loc["related_equal_mass"].related_mass
        mc = m.loc["full_hook"].control_mass - m.loc["control_equal_mass"].control_mass
        tolerance = min(1e-3, .02 * min(mg, mc))
        dose_ok = abs(mg - mc) <= tolerance
        assert mg > 0 and mc > 0 and dose_ok
        for route, untouched in (("related_equal_mass", "control_mass"), ("control_equal_mass", "related_mass")):
            assert abs(m.loc[route, untouched] - m.loc["full_hook", untouched]) <= 1e-3
        assert abs(m.loc["related_equal_mass", "receiver_mass"] - m.loc["control_equal_mass", "receiver_mass"]) <= 1e-3
        old_mass = raw_by_route["full_hook"]["weight_before"][slot, 0, case["related_positions"]].sum()
        assert abs(m.loc["related_revert", "related_mass"] - old_mass) <= 1e-3
        path_pass = ratio >= .5 and g_equal_loss > c_equal_loss + 1e-6 and dose_ok
        shape_pass = p.loc["position_template"].bce > full.bce + 1e-6
        decisions.append({"sample_index": i, "case_id": case["case_id"], "local_off_prediction": full.source_local_off_prediction,
            "full_prediction": full.prediction, "related_revert_prediction": p.loc["related_revert"].prediction,
            "related_equal_mass_prediction": p.loc["related_equal_mass"].prediction,
            "control_equal_mass_prediction": p.loc["control_equal_mass"].prediction,
            "position_template_prediction": p.loc["position_template"].prediction,
            "full_bce_gain": gain, "rollback_loss_ratio": ratio,
            "equal_mass_G_minus_C_bce": g_equal_loss - c_equal_loss,
            "template_minus_full_bce": p.loc["position_template"].bce - full.bce,
            "actual_mass_removed_G": mg, "actual_mass_removed_C": mc,
            "actual_dose_difference": mg - mc, "dose_pass": bool(dose_ok),
            "group_path_pass": bool(path_pass), "position_shape_pass": bool(shape_pass)})
    out.mkdir(parents=True, exist_ok=True)
    for name in ("predictions.csv", "group_mass.csv", "groups.json", "manifest.json", "gates.json", "complete.json"):
        shutil.copy2(source / name, out / name)
    (out / "scheduler_accounting.txt").write_text(accounting)
    pd.DataFrame(decisions).to_csv(out / "case_decisions.csv", index=False)
    result = {"passed": True, "job_id": args.job_id, "artifact_files_verified": len(complete["files_sha256"]),
              "gates_rechecked": True, "source_predictions_and_truth_rechecked": True,
              "softmax_max_abs_errors": reconstruction_errors, "cases": decisions,
              "auditor_sha256": sha256(Path(__file__))}
    (out / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
