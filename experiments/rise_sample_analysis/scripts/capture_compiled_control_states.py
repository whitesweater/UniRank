#!/usr/bin/env python3
"""Single-variant compiled capture of actual _final_channels inputs/outputs.

Instance-local wrappers only store tensor references and call the original bound
method. Statistics run after the forward. The instrumented compiled prediction
must match the uninstrumented checkpoint before any state-derived observations
are accepted. Original core/observer/diagnostic/benchmark files are unchanged.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diagnose_mechanism_precision import CompiledTensorCapture, comparison, concatenate_predictions, route_predictions
from observe_mechanism import (
    ROOT, LABELS, VARIANTS, SISAScoreBias, atomic_json, device_metadata, file_identity,
    load_model, load_sample_cache, masked_mean, parse_log, prediction_arrays,
    run_records, seed_everything, sha256, tree_map, validate_parameters,
)


def captured_state_statistics(module, query_record, key_record, valid_mask,
                              query_slice=None, key_slice=None):
    qargs, _, query_final = query_record
    kargs, _, key_final = key_record
    raw_q, query_decay, query_phase, query_offset = qargs
    raw_k, key_decay, key_phase, key_offset = kargs
    qslice, kslice = query_slice or slice(None), key_slice or slice(None)
    qvalid, kvalid = valid_mask.bool()[:, qslice], valid_mask.bool()[:, kslice]
    pair_valid = qvalid[:, None, :, None] & kvalid[:, None, None, :]
    with torch.autocast(device_type=raw_q.device.type, enabled=False):
        qcenter = query_decay - query_offset[..., None]
        kcenter = key_decay - key_offset[..., None]
        qexp = qcenter.clamp(-module.exp_clamp, module.exp_clamp)[:, :, qslice].float()
        kexp = -kcenter.clamp(-module.exp_clamp, module.exp_clamp)[:, :, kslice].float()
        log_amplitude = qexp[..., None] + kexp[:, :, None, :]
        q, k = raw_q[:, :, qslice].float(), raw_k[:, :, kslice].float()
        final_q, final_k = query_final[:, :, qslice].float(), key_final[:, :, kslice].float()
        qnorm, knorm = q.norm(dim=-1), k.norm(dim=-1)
        final_qnorm, final_knorm = final_q.norm(dim=-1), final_k.norm(dim=-1)
        original_dot = q @ k.transpose(-1, -2)
        final_dot = final_q @ final_k.transpose(-1, -2)
        norm_valid = pair_valid & (qnorm[..., None] > 0) & (knorm[:, :, None, :] > 0)
        norm_valid &= (final_qnorm[..., None] > 0) & (final_knorm[:, :, None, :] > 0)
        original_cos = original_dot / (qnorm[..., None] * knorm[:, :, None, :]).clamp_min(1e-30)
        final_cos = final_dot / (final_qnorm[..., None] * final_knorm[:, :, None, :]).clamp_min(1e-30)
        qgain = final_qnorm / qnorm.clamp_min(1e-30)
        kgain = final_knorm / knorm.clamp_min(1e-30)
        observed_gain = qgain[..., None] * kgain[:, :, None, :]
        return {
            "state_derived_query_clipping_fraction": masked_mean((qcenter[:, :, qslice].abs() > module.exp_clamp).float(), qvalid[:, None, :]),
            "state_derived_key_clipping_fraction": masked_mean((kcenter[:, :, kslice].abs() > module.exp_clamp).float(), kvalid[:, None, :]),
            "query_decay_state_mean": masked_mean(query_decay[:, :, qslice].float(), qvalid[:, None, :]),
            "key_decay_state_mean": masked_mean(key_decay[:, :, kslice].float(), kvalid[:, None, :]),
            "query_offset_mean": query_offset.float().mean(dim=1),
            "key_offset_mean": key_offset.float().mean(dim=1),
            "query_phase_state_abs_mean": masked_mean(query_phase[:, :, qslice].float().abs(), qvalid[:, None, :, None]),
            "key_phase_state_abs_mean": masked_mean(key_phase[:, :, kslice].float().abs(), kvalid[:, None, :, None]),
            "state_derived_log_amplitude_mean": masked_mean(log_amplitude, pair_valid),
            "state_derived_amplitude_mean": masked_mean(log_amplitude.exp(), pair_valid),
            "state_derived_amplification_fraction": masked_mean((log_amplitude > 0).float(), pair_valid),
            "captured_channel_norm_gain_mean": masked_mean(observed_gain, norm_valid),
            "captured_channel_cosine_change_abs_mean": masked_mean((final_cos - original_cos).abs(), norm_valid),
            "captured_alignment_sign_change_fraction": masked_mean(((original_dot * final_dot) < 0).float(), norm_valid),
            "zero_channel_pair_fraction": masked_mean((~norm_valid).float(), pair_valid),
        }


class CompiledControlStateCapture(CompiledTensorCapture):
    def __init__(self, model, expected_sites=6):
        self.entries, self.handles, self.modules = [], [], {}
        self.control_entries, self.original_methods = [], []
        for name, parent in model.named_modules():
            # OptimizedModule forwards attributes to _orig_mod. Visit the actual
            # child only, avoiding duplicate sites for compiled target attention.
            if "_orig_mod" in parent._modules:
                continue
            bias, attention = getattr(parent, "sisa_score_bias", None), getattr(parent, "dot_attention", None)
            if not isinstance(bias, SISAScoreBias) or attention is None:
                continue
            site = ".".join(part for part in name.split(".") if part != "_orig_mod")
            if site in self.modules:
                raise ValueError(f"Duplicate site {site}")
            self.modules[site] = bias
            self.handles.append(bias.register_forward_pre_hook(self._capture(site, "aux"), with_kwargs=True))
            self.handles.append(attention.register_forward_pre_hook(self._capture(site, "attention"), with_kwargs=True))
            had_instance_method = "_final_channels" in bias.__dict__
            original_instance_value = bias.__dict__.get("_final_channels")
            original_bound = bias._final_channels
            self.original_methods.append((bias, had_instance_method, original_instance_value))
            bias._final_channels = self._wrap_final(site, original_bound)
        if len(self.modules) != expected_sites:
            self.close()
            raise ValueError(f"Expected {expected_sites} sites, got {list(self.modules)}")

    def _wrap_final(self, site, original_bound):
        def wrapper(*args, **kwargs):
            output = original_bound(*args, **kwargs)
            self.control_entries.append((site, kwargs["inverse_decay"], args, kwargs, output))
            return output
        return wrapper

    def reset(self):
        super().reset()
        self.control_entries.clear()

    def collect_states(self):
        if len(self.control_entries) != 2 * len(self.modules):
            raise RuntimeError(f"Expected two _final_channels captures per site; got {len(self.control_entries)}")
        result = {}
        for site, inverse, args, kwargs, output in self.control_entries:
            label = "key" if inverse else "query"
            states = result.setdefault(site, {})
            if label in states or len(args) != 4:
                raise RuntimeError("Duplicate/malformed final-channel state capture")
            states[label] = (args, kwargs, output)
        if set(result) != set(self.modules) or any(set(states) != {"query", "key"} for states in result.values()):
            raise RuntimeError("Incomplete control state captures")
        return result

    def observation_rows(self, meta, variant):
        selection = self.selection_rows(meta, variant)
        lookup = {(row["site"], row["part_id"], row["row_offset"]): row for row in selection}
        aux = self.collect()
        for site, states in self.collect_states().items():
            args, kwargs = aux[site]["aux"]
            module = self.modules[site]
            summary = captured_state_statistics(module, states["query"], states["key"], args[1],
                                               kwargs.get("query_slice"), kwargs.get("key_slice"))
            values = {key: tensor.detach().float().cpu().numpy() for key, tensor in summary.items()}
            for i in range(len(meta["row_offset"])):
                row = lookup[(site, int(meta["part_id"][i]), int(meta["row_offset"][i]))]
                row.update({key: float(value[i]) for key, value in values.items()})
                row.update(capture_route="compiled_final_channels_inputs_outputs",
                           decay_reference=module.decay_reference, score_dim=module.score_dim,
                           control_variant=getattr(module, "control", module.variant),
                           control_state_recomputed_from_hidden=False,
                           decay_state_dtype=str(states["key"][0][1].dtype),
                           phase_state_dtype=str(states["key"][0][2].dtype),
                           offset_dtype=str(states["key"][0][3].dtype),
                           final_channel_dtype=str(states["key"][2].dtype))
        return selection

    def close(self):
        for module, had_instance_method, original_instance_value in self.original_methods:
            if had_instance_method:
                module._final_channels = original_instance_value
            else:
                delattr(module, "_final_channels")
        self.original_methods.clear()
        self.control_entries.clear()
        super().close()


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("B", "E"), required=True,
                        help="Exactly one variant per process to isolate compilation/lifetime")
    parser.add_argument("--limit-batches", type=int, default=1)
    args = parser.parse_args()
    if args.limit_batches < 1:
        parser.error("limit-batches must be positive")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1 or not torch.cuda.is_available():
        raise RuntimeError("Single GPU inference process required")
    os.chdir(ROOT)
    reference, output = args.reference_dir.resolve(), args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "artifacts/rise_sample_analysis/mechanism"):
        parser.error("Output must be under artifacts/rise_sample_analysis/mechanism")
    output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((reference / "manifest.json").read_text())
    sample_path = reference / "sample_batches.pt"
    if sha256(sample_path) != prior["sample_cache_sha256"]:
        raise ValueError("Frozen sample cache changed")
    samples = load_sample_cache(sample_path)[:args.limit_batches]
    meta = pd.concat([pd.DataFrame(item["meta"]) for item in samples], ignore_index=True)
    meta.to_parquet(output / "sample_rows.parquet", index=False)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision(prior["device"]["float32_matmul_precision"])
    for relative, expected in prior["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen original source changed: {relative}")
    paths = [Path(__file__).resolve(), Path(__file__).with_name("diagnose_mechanism_precision.py")]
    manifest = {"variant": args.variant, "rows": len(meta), "device": device_metadata(),
                "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
                "original_manifest_sha256": sha256(reference / "manifest.json"),
                "sample_cache_sha256": sha256(sample_path), "prediction_atol": prior["prediction_atol"],
                "prediction_rtol": prior["prediction_rtol"], "one_variant_per_process": True,
                "capture_definition": "instrumented compiled _final_channels actual arguments and output tensors",
                "state_metrics_derived_after_forward": True, "states_recomputed_from_hidden": False,
                "no_attention_or_state_matrices_persisted": True,
                "phase_and_decay_are_graph_states_not_physical_kernel_register_claims": True}
    atomic_json(output / "manifest.json", manifest)
    records = run_records()
    base, _ = parse_log(ROOT / records["E"]["checkpoint_log"])
    log = ROOT / records[args.variant]["checkpoint_log"]
    params, _ = parse_log(log)
    validate_parameters(params, base, args.variant)
    params.update(sisa_variant=VARIANTS[args.variant], distributed=False, world_size=1, rank=0, local_rank=0)
    checkpoint = log.with_suffix(".model")
    expected_checkpoint = json.loads((reference / f"{args.variant}_prediction_gate.json").read_text())["checkpoint_sha256"]
    if sha256(checkpoint) != expected_checkpoint:
        raise ValueError("Checkpoint differs from frozen reference")
    checkpoint_before = file_identity(checkpoint)
    seed_everything(params["seed"])
    atomic_json(output / "progress.json", {"phase": "load_compiled"})
    model = load_model(params, checkpoint, 0)
    atomic_json(output / "progress.json", {"phase": "compiled_plain"})
    plain, _ = route_predictions(model, samples)
    capture = CompiledControlStateCapture(model)
    torch._dynamo.reset()
    atomic_json(output / "progress.json", {"phase": "compiled_state_capture"})
    captured, rows = [], []
    try:
        for index, sample in enumerate(samples):
            capture.reset()
            with torch.no_grad(), model._get_amp_context():
                predictions = model.forward(tree_map(sample["batch"], lambda tensor: tensor.clone()))
            captured.append(prediction_arrays(predictions))
            with torch.no_grad():
                rows += capture.observation_rows(sample["meta"], args.variant)
            atomic_json(output / "progress.json", {"phase": "compiled_state_capture", "completed_batches": index + 1})
    finally:
        capture.close()
    captured = concatenate_predictions(captured)
    gate = comparison(captured, plain, prior["prediction_atol"], prior["prediction_rtol"])
    previous = pd.read_parquet(reference / f"{args.variant}_prediction_check.parquet")
    joined = meta.merge(previous, on=list(meta.columns), how="left", validate="one_to_one", sort=False)
    old_compiled = {label: joined[f"{label}_compiled"].to_numpy() for label in LABELS}
    gate["plain_vs_frozen_compiled"] = comparison(plain, old_compiled, prior["prediction_atol"], prior["prediction_rtol"])
    gate["accepted"] = gate["passed"] and gate["plain_vs_frozen_compiled"]["passed"]
    gate["checkpoint_sha256"] = expected_checkpoint
    atomic_json(output / "prediction_gate.json", gate)
    predictions = meta.copy()
    for label in LABELS:
        predictions[f"{label}_compiled_plain"] = plain[label]
        predictions[f"{label}_compiled_state_capture"] = captured[label]
    predictions.to_parquet(output / "prediction_check.parquet", index=False)
    observations = pd.DataFrame(rows)
    if len(observations) != 6 * len(meta) or observations.duplicated(["part_id", "row_offset", "site"]).any():
        raise ValueError("Incomplete sample/site coverage")
    suffix = "validated" if gate["accepted"] else "unvalidated"
    observations.to_parquet(output / f"{args.variant}_compiled_states_{suffix}.parquet", index=False)
    if checkpoint_before != file_identity(checkpoint):
        raise ValueError("Checkpoint changed during capture")
    if not gate["accepted"]:
        atomic_json(output / "capture_failed_gate.json", {"accepted": False, "thresholds_unchanged": True})
        raise RuntimeError("Compiled state capture changed predictions beyond the frozen gate")
    atomic_json(output / "complete.json", {"variant": args.variant, "rows": len(meta), "sites": 6,
                "prediction_gate_passed": True, "capture_prediction_bitwise_equal": gate["bitwise_equal"],
                "plain_matches_frozen_compiled_bitwise": gate["plain_vs_frozen_compiled"]["bitwise_equal"],
                "states_recomputed_from_hidden": False, "observational_only": True})


if __name__ == "__main__":
    main()
