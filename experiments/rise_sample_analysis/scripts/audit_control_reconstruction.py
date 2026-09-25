#!/usr/bin/env python3
"""Audit learned-control recomputation on actual compiled hidden representations.

This is a reconstruction, not an observation of fused AM/PA intermediate states.
Actual compiled delta is captured without arithmetic hooks; a separate eager
BF16 recomputation from the same hidden tensors/weights must pass every-position
delta and induced-selection gates before reconstructed AM/PA statistics are
eligible for interpretation. Existing scripts and failed evidence are untouched.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diagnose_mechanism_precision import CompiledTensorCapture, comparison, concatenate_predictions, route_predictions
from observe_mechanism import (
    ROOT, LABELS, VARIANTS, SISAScoreBias, atomic_json, control_statistics,
    device_metadata, file_identity, load_model, load_sample_cache, masked_mean,
    native_allowed, parse_log, prediction_arrays, run_records, seed_everything,
    sha256, tree_map, validate_parameters,
)

# These diagnostic gates are fixed before any GPU result from this script.
# Delta is measured in attention-logit units, not final prediction probability.
DELTA_ATOL = 5e-4
DELTA_RTOL = 5e-3
SELECTION_TV_ATOL = 1e-3


def reconstruction_statistics(query, key, actual_delta, reconstructed_delta, *,
                              scale=None, mask=None, is_causal=False, query_valid=None):
    if actual_delta.shape != reconstructed_delta.shape:
        raise ValueError("Actual/reconstructed delta shapes differ")
    with torch.autocast(device_type=query.device.type, enabled=False):
        actual, rebuilt = actual_delta.float(), reconstructed_delta.float()
        if not torch.isfinite(actual).all() or not torch.isfinite(rebuilt).all():
            raise ValueError("Nonfinite delta reconstruction")
        absolute = (rebuilt - actual).abs()
        failed = absolute > DELTA_ATOL + DELTA_RTOL * actual.abs()
        all_pairs = torch.ones_like(actual, dtype=torch.bool)
        allowed = native_allowed(query, key, mask, is_causal)
        any_allowed = allowed.any(-1)
        row_valid = any_allowed.clone()
        if query_valid is not None:
            row_valid &= query_valid[:, None, :]
        used_pairs = allowed & row_valid[..., None]
        axes = (1, 2, 3)
        denominator = actual.abs().masked_fill(~used_pairs, 0).sum(axes)
        numerator = absolute.masked_fill(~used_pairs, 0).sum(axes)
        relative_l1 = numerator / denominator.clamp_min(1e-12)
        logits = query.float() @ key.float().transpose(-1, -2)
        logits /= float(scale) if scale is not None else query.shape[-1] ** 0.5
        first = (logits + actual).masked_fill(~allowed, -torch.inf)
        second = (logits + rebuilt).masked_fill(~allowed, -torch.inf)
        first = torch.where(any_allowed[..., None], first, torch.zeros_like(first))
        second = torch.where(any_allowed[..., None], second, torch.zeros_like(second))
        probabilities = torch.softmax(first, dim=-1)
        reconstructed_probabilities = torch.softmax(second, dim=-1)
        tv = 0.5 * (probabilities - reconstructed_probabilities).abs().sum(-1)
        difference = rebuilt - actual
        osc = (difference.masked_fill(~allowed, -torch.inf).amax(-1)
               - difference.masked_fill(~allowed, torch.inf).amin(-1))
        osc = torch.where(any_allowed, osc, torch.zeros_like(osc))
        changed = first.argmax(-1) != second.argmax(-1)
        return {
            "delta_positions_all": all_pairs.sum(axes),
            "delta_failed_positions_all": failed.sum(axes),
            "delta_failure_fraction_all": masked_mean(failed.float(), all_pairs),
            "delta_positions_used": used_pairs.sum(axes),
            "delta_failed_positions_used": (failed & used_pairs).sum(axes),
            "delta_failure_fraction_used": masked_mean(failed.float(), used_pairs),
            "delta_abs_error_max_all": absolute.amax(axes),
            "delta_abs_error_mean_used": masked_mean(absolute, used_pairs),
            "delta_relative_l1_used": relative_l1,
            "delta_bitwise_equal_per_example": (actual == rebuilt).all(dim=3).all(dim=2).all(dim=1),
            "reconstruction_tv_mean": masked_mean(tv, row_valid),
            "reconstruction_tv_max": tv.masked_fill(~row_valid, 0).amax((1, 2)),
            "reconstruction_tv_failed_rows": ((tv > SELECTION_TV_ATOL) & row_valid).sum((1, 2)),
            "reconstruction_topkey_changed_fraction": masked_mean(changed.float(), row_valid),
            "reconstruction_tv_error_bound_mean": masked_mean(torch.tanh(osc / 4), row_valid),
            "reconstruction_valid_attention_rows": row_valid.sum((1, 2)),
        }


def reconstruction_gate(frame):
    failed_positions = int(frame["delta_failed_positions_all"].sum())
    failed_rows = int(frame["reconstruction_tv_failed_rows"].sum())
    return {
        "delta_atol_logit_units": DELTA_ATOL, "delta_rtol": DELTA_RTOL,
        "selection_tv_atol": SELECTION_TV_ATOL,
        "positions_checked": int(frame["delta_positions_all"].sum()),
        "failed_positions": failed_positions,
        "attention_rows_checked": int(frame["reconstruction_valid_attention_rows"].sum()),
        "failed_attention_rows": failed_rows,
        "max_delta_abs_error": float(frame["delta_abs_error_max_all"].max()),
        "max_reconstruction_tv": float(frame["reconstruction_tv_max"].max()),
        "all_delta_positions_passed": failed_positions == 0,
        "all_selection_rows_passed": failed_rows == 0,
        "passed": failed_positions == 0 and failed_rows == 0,
        "does_not_identify_actual_fused_intermediate_states": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="Original observed_643858 directory holding the fixed input cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=("B", "E"), default=["B", "E"])
    parser.add_argument("--limit-batches", type=int)
    args = parser.parse_args()
    if args.limit_batches is not None and args.limit_batches < 1:
        parser.error("limit-batches must be positive")
    if len(args.variants) != len(set(args.variants)):
        parser.error("duplicate variants")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1 or not torch.cuda.is_available():
        raise RuntimeError("Single GPU inference process required")
    os.chdir(ROOT)
    output, reference = args.output_dir.resolve(), args.reference_dir.resolve()
    if not output.is_relative_to(ROOT / "artifacts/rise_sample_analysis/mechanism"):
        parser.error("Output must be under artifacts/rise_sample_analysis/mechanism")
    output.mkdir(parents=True, exist_ok=False)
    previous_manifest = json.loads((reference / "manifest.json").read_text())
    sample_path = reference / "sample_batches.pt"
    if sha256(sample_path) != previous_manifest["sample_cache_sha256"]:
        raise ValueError("Reference sample cache changed")
    samples = load_sample_cache(sample_path)
    if args.limit_batches is not None:
        samples = samples[:args.limit_batches]
    meta = pd.concat([pd.DataFrame(sample["meta"]) for sample in samples], ignore_index=True)
    meta.to_parquet(output / "sample_rows.parquet", index=False)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision(previous_manifest["device"]["float32_matmul_precision"])
    source_hashes = dict(previous_manifest["source_sha256"])
    for path in (Path(__file__).resolve(), Path(__file__).with_name("diagnose_mechanism_precision.py")):
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)
    # Reverify the old observation/core/export source contract, rather than
    # silently accepting a simultaneously modified reconstruction definition.
    for relative, expected in previous_manifest["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Original source changed: {relative}")
    manifest = {
        "reference_dir": str(reference), "reference_manifest_sha256": sha256(reference / "manifest.json"),
        "source_sha256": source_hashes, "device": device_metadata(), "rows": len(meta),
        "variants": args.variants, "limit_batches": args.limit_batches,
        "sample_cache_sha256": sha256(sample_path),
        "reconstruction_definition": "original eager BF16 SISA forward on captured compiled hidden vectors, masks and checkpoint weights",
        "interpretation": "learned-control reconstruction; never literal fused AM/PA intermediate states",
        "delta_atol_logit_units": DELTA_ATOL, "delta_rtol": DELTA_RTOL,
        "selection_tv_atol": SELECTION_TV_ATOL,
        "prediction_atol": previous_manifest["prediction_atol"],
        "prediction_rtol": previous_manifest["prediction_rtol"],
        "all_statistics_after_model_forward": True, "no_attention_matrices_persisted": True,
    }
    atomic_json(output / "manifest.json", manifest)
    records = run_records()
    base, _ = parse_log(ROOT / records["E"]["checkpoint_log"])
    statuses = {}
    for variant in args.variants:
        log = ROOT / records[variant]["checkpoint_log"]
        params, _ = parse_log(log)
        validate_parameters(params, base, variant)
        params.update(sisa_variant=VARIANTS[variant], distributed=False, world_size=1, rank=0, local_rank=0)
        checkpoint = log.with_suffix(".model")
        previous_gate = json.loads((reference / f"{variant}_prediction_gate.json").read_text())
        if sha256(checkpoint) != previous_gate["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed")
        checkpoint_before = file_identity(checkpoint)
        seed_everything(params["seed"])
        torch._dynamo.reset()
        model = load_model(params, checkpoint, 0)
        plain, _ = route_predictions(model, samples)
        capture = CompiledTensorCapture(model)
        capture.variant = variant
        torch._dynamo.reset()
        captured, rows = [], []
        try:
            for sample in samples:
                capture.reset()
                with torch.no_grad(), model._get_amp_context():
                    predictions = model.forward(tree_map(sample["batch"], lambda t: t.clone()))
                captured.append(prediction_arrays(predictions))
                for site, item in capture.collect().items():
                    aux_args, aux_kwargs = item["aux"]
                    att_args, att_kwargs = item["attention"]
                    module = capture.modules[site]
                    # Call forward directly to avoid running the capture hooks
                    # a second time during this independent recomputation.
                    with torch.no_grad(), model._get_amp_context():
                        reconstructed = SISAScoreBias.forward(module, *aux_args, **aux_kwargs)
                        controls, qvalid, _ = control_statistics(module, *aux_args, **aux_kwargs)
                    audit = reconstruction_statistics(att_args[0], att_args[1], att_kwargs["score_bias"],
                        reconstructed, scale=att_kwargs.get("scale"), mask=att_kwargs.get("mask"),
                        is_causal=att_kwargs.get("is_causal", False), query_valid=qvalid)
                    values = {key: value.detach().float().cpu().numpy()
                              for key, value in (controls | audit).items()}
                    for i in range(len(sample["meta"]["row_offset"])):
                        row = {key: int(value[i]) for key, value in sample["meta"].items()}
                        row.update(variant=variant, site=site,
                            reconstruction="eager_bf16_from_compiled_hidden", decay_reference=module.decay_reference,
                            actual_delta_dtype=str(att_kwargs["score_bias"].dtype),
                            reconstructed_delta_dtype=str(reconstructed.dtype),
                            actual_fused_internal_state_observed=False)
                        row.update({key: float(value[i]) for key, value in values.items()})
                        rows.append(row)
        finally:
            capture.close()
        captured = concatenate_predictions(captured)
        pred_gate = comparison(captured, plain, previous_manifest["prediction_atol"], previous_manifest["prediction_rtol"])
        atomic_json(output / f"{variant}_capture_prediction_gate.json", pred_gate)
        frame = pd.DataFrame(rows)
        if len(frame) != len(meta) * 6 or frame.duplicated(["part_id", "row_offset", "site"]).any():
            raise ValueError("Incomplete reconstructed sample/site coverage")
        gate = reconstruction_gate(frame)
        gate["capture_prediction_gate_passed"] = pred_gate["passed"]
        gate["eligible_reconstruction"] = pred_gate["passed"] and gate["passed"]
        gate["site_gates"] = {site: reconstruction_gate(group) for site, group in frame.groupby("site")}
        atomic_json(output / f"{variant}_reconstruction_gate.json", gate)
        suffix = "validated" if gate["eligible_reconstruction"] else "unvalidated"
        frame.to_parquet(output / f"{variant}_learned_control_reconstruction_{suffix}.parquet", index=False)
        pred = meta.copy()
        for label in LABELS:
            pred[f"{label}_compiled_plain"] = plain[label]
            pred[f"{label}_compiled_capture"] = captured[label]
        pred.to_parquet(output / f"{variant}_prediction_check.parquet", index=False)
        statuses[variant] = {"capture_passed": pred_gate["passed"],
                             "capture_bitwise_equal": pred_gate["bitwise_equal"],
                             "eligible_reconstruction": gate["eligible_reconstruction"]}
        if checkpoint_before != file_identity(checkpoint):
            raise ValueError("Checkpoint changed during reconstruction audit")
        del model, capture, frame
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(output / "audit_complete.json", {
        "audit_execution_completed": True, "variants": statuses,
        "all_reconstructions_eligible": all(row["eligible_reconstruction"] for row in statuses.values()),
        "actual_fused_am_pa_internal_state_observed": False,
        "failed_numeric_gates_are_audit_results_not_successful_reconstructions": True,
    })


if __name__ == "__main__":
    main()
