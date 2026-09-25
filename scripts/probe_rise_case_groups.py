"""Frozen KuaiRand semantic-group mass interventions; diagnostic inference only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_rise_semantic_attention as semantic

engine = semantic.engine
ROUTES = ("related_revert", "related_equal_mass", "control_equal_mass", "position_template")


def checked_positions(positions, allowed):
    """Require a nonempty proper subset of the native allowed keys."""
    allowed = np.asarray(allowed, dtype=bool)
    if not isinstance(positions, (list, tuple, np.ndarray)) or not len(positions):
        raise ValueError("A group must contain at least one position")
    if any(isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer)) for p in positions):
        raise ValueError("Group positions must be zero-based integers")
    positions = np.asarray(positions, dtype=np.int64)
    if (len(np.unique(positions)) != len(positions) or np.any(positions < 0)
            or np.any(positions >= len(allowed)) or not np.all(allowed[positions])
            or len(positions) >= int(allowed.sum())):
        raise ValueError("Group must be unique, valid, and leave an allowed complement")
    return positions


def mass_shift(current_mass, target_mass):
    """Constant score shift giving target group mass in exact arithmetic."""
    if (not np.isfinite([current_mass, target_mass]).all()
            or not 0 < current_mass < 1 or not 0 < target_mass < 1):
        raise ValueError("Current and target mass must both lie strictly inside (0,1)")
    return float(np.log(target_mass) - np.log1p(-target_mass)
                 - np.log(current_mass) + np.log1p(-current_mass))


def group_interventions(before, after, related, control, allowed):
    """Build gauge-invariant group shifts and a shared feasible mass dose."""
    related = checked_positions(related, allowed)
    control = checked_positions(control, allowed)
    if np.intersect1d(related, control).size:
        raise ValueError("Related and control groups must be disjoint")
    before, after = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    for value in (before, after):
        if (value.shape != np.asarray(allowed).shape or not np.isfinite(value).all()
                or np.any(value < 0) or not np.isclose(value.sum(), 1, atol=2e-6, rtol=0)
                or np.any(value[~np.asarray(allowed, dtype=bool)] != 0)):
            raise ValueError("Expected normalized finite probabilities on the native allowed set")
    m0, m1 = float(before[related].sum()), float(after[related].sum())
    mc = float(after[control].sum())
    receiver = np.flatnonzero(np.asarray(allowed, dtype=bool)
                             & ~np.isin(np.arange(len(after)), np.concatenate((related, control))))
    mr = float(after[receiver].sum())
    if not len(receiver) or not 0 < mr < 1:
        raise ValueError("A nonempty positive-mass common receiver outside both groups is required")
    lift = m1 - m0
    if lift <= 0:
        raise ValueError("The related group must receive positive mass lift")
    dose = min(lift, 0.9 * mc)
    if dose <= 0:
        raise ValueError("No positive feasible shared mass dose")
    adjustments = {route: np.zeros_like(after) for route in ROUTES if route != "position_template"}
    adjustments["related_revert"][related] = mass_shift(m1, m0)
    for name, positions, current in (("related_equal_mass", related, m1), ("control_equal_mass", control, mc)):
        adjustments[name][positions] = np.log((current - dose) / current)
        adjustments[name][receiver] = np.log((mr + dose) / mr)
    return adjustments, {"related_mass_before": m0, "related_mass_full": m1,
                         "control_mass_full": mc, "receiver_mass_full": mr, "related_mass_lift": lift,
                         "equal_mass_requested_dose": dose,
                         "ideal_target_mass": {"related_revert": m0, "related_equal_mass": m1 - dose,
                                               "control_equal_mass": mc - dose},
                         "equal_mass_receiver_target": mr + dose,
                         "equal_mass_receiver_positions": receiver.tolist(),
                         "equal_mass_untouched_group_fixed": True,
                         "score_shift": {name: {"related": float(shifts[related[0]]),
                                                "control": float(shifts[control[0]]),
                                                "receiver": float(shifts[receiver[0]])}
                                         for name, shifts in adjustments.items()},
                         "related_count": len(related), "control_count": len(control),
                         "related_mean_position": float(related.mean()),
                         "control_mean_position": float(control.mean()),
                         "mean_position_difference": float(related.mean() - control.mean()),
                         "strict_position_matching_claimed": False}


def position_template(deltas, masks, sample_index):
    """LOO mean position bias, using only identical allowed sets, RMS matched."""
    deltas, masks = np.asarray(deltas, dtype=np.float64), np.asarray(masks, dtype=bool)
    if deltas.ndim != 2 or masks.shape != deltas.shape or not 0 <= sample_index < len(deltas):
        raise ValueError("Invalid position-template source arrays")
    allowed = masks[sample_index]
    peers = np.flatnonzero(np.all(masks == allowed, axis=1) & (np.arange(len(masks)) != sample_index))
    if not allowed.any() or not len(peers):
        raise ValueError("Position template requires nonempty history and same-mask LOO peers")
    source = deltas[sample_index, allowed]
    template = deltas[peers][:, allowed].mean(axis=0)
    if not np.isfinite(source).all() or not np.isfinite(template).all():
        raise ValueError("Nonfinite position-template data")
    source_mean = float(source.mean())
    source_rms = float(np.sqrt(np.mean((source - source_mean) ** 2)))
    centered = template - template.mean()
    template_rms = float(np.sqrt(np.mean(centered ** 2)))
    if template_rms <= 0 and source_rms > 0:
        raise ValueError("Flat position template cannot match positive source RMS")
    matched = np.zeros_like(centered) if source_rms == 0 else centered * (source_rms / template_rms)
    result = deltas[sample_index].copy()
    result[allowed] = matched + source_mean
    return result, {"peer_indices": peers.tolist(), "peer_count": len(peers),
                    "same_valid_mask_required": True, "excluded_sample_index": int(sample_index),
                    "source_mean": source_mean, "source_centered_rms": source_rms,
                    "template_centered_rms_before_rescaling": template_rms,
                    "template_mean_before_cast": float(result[allowed].mean()),
                    "template_centered_rms_before_cast": float(np.sqrt(np.mean(matched ** 2)))}


class BiasAdjustment(semantic.Capture):
    """Observe the actual modified bias, preserving the three upstream Q/K/V."""

    def __init__(self, model, adjustment):
        self.adjustment = adjustment
        super().__init__(model)

    def hook(self, module, args, kwargs):
        bias = kwargs.get("score_bias")
        if bias is None or len(args) != 3 or bias.shape != self.adjustment.shape:
            raise ValueError("Unexpected target-site bias shape or Q/K/V arguments")
        modified = (bias.float() + self.adjustment).to(bias.dtype)
        changed = dict(kwargs, score_bias=modified)
        self.entries.append((args, changed))
        return args, changed


def run_route(params, checkpoint, samples, route, adjustments, output):
    """Always forward original full windows; return all predictions for exact gates."""
    engine.seed_everything(params["seed"])
    model = semantic.load_model(params, checkpoint, 0)
    observer = None
    if route != "plain":
        initial = adjustments[0]
        device = next(model.parameters(), torch.empty(0)).device
        observer = BiasAdjustment(model, torch.as_tensor(initial, dtype=torch.float32, device=device).clone())
        torch._dynamo.reset()
    whole_predictions, raw, signatures = [], [], []
    site_info = None
    try:
        for batch_number, sample in enumerate(samples):
            if observer is not None:
                observer.adjustment.copy_(torch.as_tensor(adjustments[batch_number], device=observer.adjustment.device))
            with torch.no_grad(), model._get_amp_context():
                values = model.forward(semantic.tree_map(sample["batch"], lambda tensor: tensor.clone()))
            predictions = {label: values[f"{label}_pred"].detach().float().cpu().numpy().reshape(-1)
                           for label in semantic.LABELS}
            if any(not np.isfinite(p).all() for p in predictions.values()):
                raise ValueError("Nonfinite route prediction")
            whole_predictions.append(predictions)
            if observer is not None:
                raw.append(observer.consume(sample["selected_indices"]))
                signatures.append(observer.qkv_signatures[-1])
                site_info = observer.site_info
            engine.atomic_json(output / "progress.json", {"phase": route, "windows_complete": batch_number + 1})
            logging.info("%s: complete %s/%s original windows", route, batch_number + 1, len(samples))
    finally:
        if observer is not None:
            observer.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return whole_predictions, raw, {"route": route, "qkv_signatures": signatures, "site_info": site_info,
        "full_forward_rows": sum(len(p[semantic.LABELS[0]]) for p in whole_predictions),
        "full_prediction_sha256": [{label: hashlib.sha256(p[label].tobytes()).hexdigest()
                                     for label in semantic.LABELS} for p in whole_predictions]}


def validate_cases(document, raw):
    cases = document.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 3:
        raise ValueError("Expected between one and three fixed cases")
    indices = [case.get("sample_index") for case in cases]
    if (any(type(i) is not int or not 0 <= i < 512 for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("Cases must have distinct existing sample_index values in [0,511]")
    plans, receipts = {}, []
    for case in cases:
        index = case["sample_index"]
        plans[index], receipt = group_interventions(raw["weight_before"][index, 0],
            raw["weight_after"][index, 0], case["related_positions"], case["control_positions"], raw["valid_mask"][index])
        template, template_info = position_template(raw["delta"][:, 0], raw["valid_mask"], index)
        # FP32 addition followed by a single cast to native BF16 occurs in the hook.
        plans[index]["position_template"] = template - raw["delta"][index, 0]
        receipts.append(dict(case, **receipt, position_template=template_info))
    return cases, plans, receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-json", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "artifacts/rise_attention_semantic/kuairand_646410")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise RuntimeError("Requires one allocated GPU and one process")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "artifacts/rise_attention_case_search"):
        raise ValueError("Output must be inside artifacts/rise_attention_case_search")
    output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((source / "manifest.json").read_text())
    completed = json.loads((source / "complete.json").read_text())
    if not completed.get("completed") or prior["examples"] != 512:
        raise ValueError("Source diagnostic is not a complete 512-case export")
    watched = {source / name: digest for name, digest in completed["files_sha256"].items()}
    watched.update({ROOT / path: digest for path, digest in prior["source_sha256"].items()})
    for name, identity in prior["input_identities"].items():
        watched[Path(identity["path"])] = prior["input_sha256"][name]
    checkpoint = Path(prior["checkpoint"]["path"])
    watched[checkpoint] = prior["checkpoint_sha256"]
    watched[args.groups_json.resolve()] = engine.sha256(args.groups_json)
    protocol = ROOT / "experiments/rise_attention_case_evidence/protocol.md"
    own_sources = (Path(__file__), ROOT / "scripts/submit_rise_case_groups.sbatch",
                   ROOT / "tests/test_rise_case_groups.py", protocol)
    for path in own_sources:
        watched[path] = engine.sha256(path)
    for path, digest in watched.items():
        if engine.sha256(path) != digest:
            raise ValueError(f"Frozen source/input hash mismatch: {path}")
    for path, identity in prior["data_files"].items():
        if engine.file_identity(Path(path)) != identity:
            raise ValueError(f"Original data identity changed: {path}")
    identities = {str(path): engine.file_identity(path) for path in watched}
    params = prior["training_params"]
    if (params["batch_size"] != 16384 or params["max_len"] != 100 or not params["enable_bf16"]
            or not params["enable_torch_compile"] or prior["site_info"]["num_heads"] != 1):
        raise ValueError("Expected original batch 16384, length 100, one-head compiled BF16 inference")
    engine.DATASET, engine.LABELS = semantic.DATASET, semantic.LABELS
    raw_file = np.load(source / "raw_attention.npz")
    raw = {key: raw_file[key] for key in raw_file.files}
    document = json.loads(args.groups_json.read_text())
    cases, plans, receipts = validate_cases(document, raw)
    source_meta = pd.read_csv(source / "meta.csv")
    source_predictions = pd.read_csv(source / "predictions.csv", float_precision="round_trip")
    cached = torch.load(source / "sample_batches.pt", map_location="cpu", weights_only=True, mmap=True)
    samples = [{"batch": item["batch"], "selected_indices": item["selected_indices"].numpy(),
                "meta": {key: value.numpy() for key, value in item["meta"].items()}} for item in cached]
    actual_meta = pd.concat([pd.DataFrame(sample["meta"]) for sample in samples], ignore_index=True)
    actual_meta.insert(0, "sample_index", np.arange(len(actual_meta)))
    pd.testing.assert_frame_equal(actual_meta, source_meta, check_dtype=False)
    if len(samples) != 8 or any(sample["batch"][2].shape != (16384, 100) for sample in samples):
        raise ValueError("Frozen cache full-window layout changed")
    manifest = {"schema_version": 1, "source_dir": str(source), "site": semantic.SITE,
        "model": semantic.MODEL, "dataset": semantic.DATASET, "training_params": params,
        "groups_document": document, "case_plans": receipts, "routes": ["plain", "full_hook", *ROUTES],
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "input_source_sha256": {str(path): digest for path, digest in watched.items()},
        "input_identities": identities, "device": semantic.device_metadata(),
        "checkpoint_sha256": prior["checkpoint_sha256"], "checkpoint": prior["checkpoint"],
        "reconstruction": "FP32 softmax from actual compiled Q/K and actual BF16 modified bias; native allowed set",
        "scope": "Same-checkpoint group mass sensitivity; no additive per-key attribution or population claim",
        "sampling": "Reuse original 512 cases and original full 16384-row windows; no new sampling or training",
        "position_control": "Label-free leave-one-out mean position correction from same-mask existing samples; match current centered RMS and mean before BF16 cast",
        "gate_tolerance": {"atol": 0, "rtol": 0}}
    for path in own_sources:
        target = output / "frozen_sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    (output / "groups.json").write_bytes(args.groups_json.read_bytes())
    engine.atomic_json(output / "manifest.json", manifest)
    zeros = [np.zeros((16384, 1, 1, 100), dtype=np.float32) for _ in samples]
    plain, _, plain_info = run_route(params, checkpoint, samples, "plain", None, output)
    selected_plain = {label: np.concatenate([p[label][s["selected_indices"]]
                        for p, s in zip(plain, samples)]) for label in semantic.LABELS}
    reference = {label: source_predictions[f"{label}_plain"].to_numpy(dtype=np.float32) for label in semantic.LABELS}
    gates = {"plain_vs_source512": semantic.prediction_comparison(selected_plain, reference)}
    engine.atomic_json(output / "gates.json", gates)
    if not gates["plain_vs_source512"]["passed"]:
        raise RuntimeError("Plain predictions differ from original 512 rows; reject probe")
    full, full_raw, full_info = run_route(params, checkpoint, samples, "full_hook", zeros, output)
    gates["full_hook"] = {
        "all_window_predictions_equal": full_info["full_prediction_sha256"] == plain_info["full_prediction_sha256"],
        "qkv_equal_source": full_info["qkv_signatures"] == prior["route_info"]["capture"]["selected_qkv_signatures_by_batch"],
        "raw_equal_source": all(np.array_equal(np.concatenate([piece[key] for piece in full_raw]), raw[key], equal_nan=True)
                                for key in full_raw[0])}
    gates["full_hook"]["passed"] = all(gates["full_hook"].values())
    engine.atomic_json(output / "gates.json", gates)
    if not gates["full_hook"]["passed"]:
        raise RuntimeError("Zero-adjustment full hook differs from original predictions/QKV/attention")
    case_indices = [case["sample_index"] for case in cases]
    active_batches = sorted({index // 64 for index in case_indices})
    all_rows, group_rows = [], []
    route_infos = {"plain": plain_info, "full_hook": full_info}
    def save_case_rows(route, route_predictions, route_raw, batch_numbers):
        for case in cases:
            index = case["sample_index"]
            batch_number, local = divmod(index, 64)
            slot = batch_numbers.index(batch_number)
            row = int(samples[batch_number]["selected_indices"][local])
            weights = route_raw[slot]["weight_after"][local, 0]
            bias = route_raw[slot]["delta"][local, 0]
            allowed = raw["valid_mask"][index]
            receiver = np.flatnonzero(allowed & ~np.isin(np.arange(len(allowed)),
                case["related_positions"] + case["control_positions"]))
            record = {"sample_index": index, "case_id": case.get("case_id", str(index)), "route": route,
                "related_mass": float(weights[case["related_positions"]].sum()),
                "control_mass": float(weights[case["control_positions"]].sum()),
                "receiver_mass": float(weights[receiver].sum()),
                "actual_bias_mean": float(bias[allowed].mean()),
                "actual_bias_centered_rms": float(np.sqrt(np.mean((bias[allowed] - bias[allowed].mean()) ** 2))),
                "all_mass": float(weights.sum())}
            group_rows.append(record)
            for label in semantic.LABELS:
                truth = int(source_meta.iloc[index][label])
                probability = float(route_predictions[slot][label][row])
                all_rows.append(dict(record, label=label, truth=truth, prediction=probability,
                    bce=float(semantic.binary_cross_entropy(np.asarray([truth]), np.asarray([probability]))[0]),
                    source_local_off_prediction=float(source_predictions.iloc[index][f"{label}_local_off"]),
                    source_local_off_bce=float(source_predictions.iloc[index][f"{label}_bce_local_off"])))
        selected_pieces = {key: np.stack([route_raw[batch_numbers.index(i // 64)][key][i % 64] for i in case_indices])
                           for key in route_raw[0]}
        np.savez_compressed(output / f"attention_{route}.npz", sample_indices=np.asarray(case_indices), **selected_pieces)
        pd.DataFrame(all_rows).to_csv(output / "predictions.csv", index=False)
        pd.DataFrame(group_rows).to_csv(output / "group_mass.csv", index=False)
    save_case_rows("full_hook", full, full_raw, list(range(8)))
    for route in ROUTES:
        adjustments = []
        for batch_number in active_batches:
            adjustment = np.zeros((16384, 1, 1, 100), dtype=np.float32)
            for index in case_indices:
                if index // 64 == batch_number:
                    row = samples[batch_number]["selected_indices"][index % 64]
                    adjustment[row, 0, 0] = plans[index][route]
            adjustments.append(adjustment)
        pred, arrays, info = run_route(params, checkpoint, [samples[b] for b in active_batches], route, adjustments, output)
        unchanged, qkv = [], []
        for slot, batch_number in enumerate(active_batches):
            keep = np.ones(16384, dtype=bool)
            for index in case_indices:
                if index // 64 == batch_number:
                    keep[samples[batch_number]["selected_indices"][index % 64]] = False
            comparison = semantic.prediction_comparison({label: p[keep] for label, p in pred[slot].items()},
                                                       {label: p[keep] for label, p in plain[batch_number].items()})
            unchanged.append(dict(comparison, batch_number=batch_number, compared_rows=int(keep.sum())))
            qkv.append(info["qkv_signatures"][slot] == full_info["qkv_signatures"][batch_number])
        gates[route] = {"nonprobe_predictions": unchanged, "all_selected_qkv_equal_full": all(qkv),
                        "passed": all(g["passed"] for g in unchanged) and all(qkv)}
        route_infos[route] = dict(info, original_batch_numbers=active_batches,
                                unexecuted_windows="Unchanged; no probe rows, predictions inherited from plain")
        engine.atomic_json(output / "gates.json", gates)
        if not gates[route]["passed"]:
            raise RuntimeError(f"{route}: nonprobe predictions or Q/K/V changed; reject probe")
        save_case_rows(route, pred, arrays, active_batches)
    for path, digest in watched.items():
        if engine.file_identity(path) != identities[str(path)] or engine.sha256(path) != digest:
            raise RuntimeError(f"Frozen source/input changed during inference: {path}")
    manifest["route_info"] = route_infos
    engine.atomic_json(output / "manifest.json", manifest)
    engine.atomic_json(output / "complete.json", {"completed": True, "all_gates_passed": True,
        "cases": case_indices, "files_sha256": {str(path.relative_to(output)): engine.sha256(path)
                                                 for path in sorted(output.rglob("*")) if path.is_file()}})


if __name__ == "__main__":
    main()
