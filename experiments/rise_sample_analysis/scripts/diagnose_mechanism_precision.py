#!/usr/bin/env python3
"""Diagnose compiled/eager and observer effects without changing old evidence.

Four routes use the same checkpoint and cached examples: compiled plain,
compiled tensor-capture only, eager plain, eager with the original observer.
The original probability tolerances are read from the failed run's manifest.
Only actual compiled Q/K/delta selection statistics are candidates for recovery;
AM/PA internal states are not claimed recovered by this diagnostic.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from observe_mechanism import (
    ROOT, LABELS, VARIANTS, MechanismObserver, SISAScoreBias, atomic_json,
    device_metadata, file_identity, load_eager_model, load_model, load_sample_cache,
    parse_log, prediction_arrays, prediction_gate, run_records, seed_everything,
    selection_statistics, sha256, tree_map, validate_parameters,
)


class CompiledTensorCapture:
    """Hooks store tensor references only; no math, CPU copies or NumPy in graph.

    Python container side effects may be lifted to outputs by Dynamo. This must
    be verified numerically on the target compiled model; it is not assumed.
    """

    def __init__(self, model, expected_sites=6):
        self.entries, self.handles, self.modules = [], [], {}
        for name, parent in model.named_modules():
            bias = getattr(parent, "sisa_score_bias", None)
            attention = getattr(parent, "dot_attention", None)
            if isinstance(bias, SISAScoreBias) and attention is not None:
                site = ".".join(part for part in name.split(".") if part != "_orig_mod")
                if site in self.modules:
                    raise ValueError(f"Duplicate canonical site {site}")
                self.modules[site] = bias
                self.handles.append(bias.register_forward_pre_hook(self._capture(site, "aux"), with_kwargs=True))
                self.handles.append(attention.register_forward_pre_hook(self._capture(site, "attention"), with_kwargs=True))
        if len(self.modules) != expected_sites:
            self.close()
            raise ValueError(f"Expected {expected_sites} sites, got {list(self.modules)}")

    def _capture(self, site, kind):
        def hook(module, args, kwargs):
            self.entries.append((site, kind, args, kwargs))
        return hook

    def reset(self):
        self.entries.clear()

    def collect(self):
        if len(self.entries) != 2 * len(self.modules):
            raise RuntimeError(f"Compiled hooks produced {len(self.entries)} captures, expected {2 * len(self.modules)}")
        result = {}
        for site, kind, args, kwargs in self.entries:
            item = result.setdefault(site, {})
            if kind in item:
                raise RuntimeError(f"Duplicate compiled capture: {site}/{kind}")
            item[kind] = (args, kwargs)
        if set(result) != set(self.modules) or any(set(item) != {"aux", "attention"} for item in result.values()):
            raise RuntimeError("Incomplete compiled site captures")
        return result

    def selection_rows(self, meta, variant):
        rows = []
        for site, item in self.collect().items():
            aux_args, aux_kwargs = item["aux"]
            args, kwargs = item["attention"]
            valid = aux_args[1].bool()
            qslice = aux_kwargs.get("query_slice") or slice(None)
            kslice = aux_kwargs.get("key_slice") or slice(None)
            stats = selection_statistics(args[0], args[1], kwargs.get("score_bias"),
                scale=kwargs.get("scale"), mask=kwargs.get("mask"),
                is_causal=kwargs.get("is_causal", False),
                query_valid=valid[:, qslice], key_valid=valid[:, kslice])
            values = {key: value.detach().float().cpu().numpy() for key, value in stats.items()}
            for i in range(len(meta["row_offset"])):
                row = {key: int(value[i]) for key, value in meta.items()}
                row.update(site=site, variant=variant, capture_route="compiled_tensor_capture",
                           native_mask_present=kwargs.get("mask") is not None,
                           native_q_dtype=str(args[0].dtype), native_k_dtype=str(args[1].dtype),
                           delta_dtype=str(kwargs["score_bias"].dtype))
                row.update({key: float(value[i]) for key, value in values.items()})
                rows.append(row)
        return rows

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.entries.clear()


def concatenate_predictions(collected):
    return {label: np.concatenate([batch[label] for batch in collected]) for label in LABELS}


def route_predictions(model, samples, observer=None):
    predictions, rows = [], []
    for sample in samples:
        if isinstance(observer, CompiledTensorCapture):
            observer.reset()
        elif isinstance(observer, MechanismObserver):
            observer.meta = sample["meta"]
        with torch.no_grad(), model._get_amp_context():
            result = model.forward(tree_map(sample["batch"], lambda tensor: tensor.clone()))
        predictions.append(prediction_arrays(result))
        if isinstance(observer, CompiledTensorCapture):
            # Run reductions after the model has completed, outside its graph.
            rows += observer.selection_rows(sample["meta"], observer.variant)
    return concatenate_predictions(predictions), rows


def comparison(left, right, atol, rtol):
    result = prediction_gate(left, right, atol, rtol)
    result["bitwise_equal"] = all(np.array_equal(left[label], right[label]) for label in LABELS)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=("B", "E"), default=["B", "E"])
    parser.add_argument("--limit-batches", type=int)
    args = parser.parse_args()
    if args.limit_batches is not None and args.limit_batches < 1:
        parser.error("limit-batches must be positive")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1 or not torch.cuda.is_available():
        raise RuntimeError("Single GPU diagnostic process required")
    os.chdir(ROOT)
    args.failed_dir = args.failed_dir.resolve()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "artifacts/rise_sample_analysis/mechanism"):
        parser.error("New output must be below artifacts/rise_sample_analysis/mechanism")
    output.mkdir(parents=True, exist_ok=False)
    failed_manifest = json.loads((args.failed_dir / "manifest.json").read_text())
    observer_source = Path(__file__).with_name("observe_mechanism.py")
    if sha256(observer_source) != failed_manifest["source_sha256"][str(observer_source.relative_to(ROOT))]:
        raise ValueError("Original observer source changed; diagnostic cannot silently change both routes")
    sample_path = args.failed_dir / "sample_batches.pt"
    if sha256(sample_path) != failed_manifest["sample_cache_sha256"]:
        raise ValueError("Failed-run sample cache changed")
    samples = load_sample_cache(sample_path)
    if args.limit_batches is not None:
        samples = samples[:args.limit_batches]
    atol, rtol = failed_manifest["prediction_atol"], failed_manifest["prediction_rtol"]
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision(failed_manifest["device"]["float32_matmul_precision"])
    manifest = {"failed_dir": str(args.failed_dir), "failed_manifest_sha256": sha256(args.failed_dir / "manifest.json"),
                "sample_cache_sha256": sha256(sample_path), "limit_batches": args.limit_batches,
                "original_atol_unchanged": atol, "original_rtol_unchanged": rtol,
                "device": device_metadata(), "script_sha256": sha256(Path(__file__)),
                "original_observer_sha256": sha256(observer_source),
                "routes": ["compiled_plain", "compiled_tensor_capture", "eager_plain", "eager_original_observer"],
                "compiler_cache_reset_after_hook_registration": True,
                "compiled_selection_capture_only": True, "am_pa_internal_states_not_recovered": True}
    atomic_json(output / "manifest.json", manifest)
    records = run_records()
    base, _ = parse_log(ROOT / records["E"]["checkpoint_log"])
    meta = pd.concat([pd.DataFrame(sample["meta"]) for sample in samples], ignore_index=True)
    meta.to_parquet(output / "sample_rows.parquet", index=False)
    summaries = {}
    for variant in args.variants:
        log = ROOT / records[variant]["checkpoint_log"]
        params, _ = parse_log(log)
        validate_parameters(params, base, variant)
        params.update(sisa_variant=VARIANTS[variant], distributed=False, world_size=1, rank=0, local_rank=0)
        checkpoint = log.with_suffix(".model")
        checkpoint_before = file_identity(checkpoint)
        previous_gate = json.loads((args.failed_dir / f"{variant}_prediction_gate.json").read_text())
        if sha256(checkpoint) != previous_gate["checkpoint_sha256"]:
            raise ValueError("Checkpoint differs from failed attempt")
        seed_everything(params["seed"])
        model = load_model(params, checkpoint, 0)
        compiled_plain, _ = route_predictions(model, samples)
        capture = CompiledTensorCapture(model)
        capture.variant = variant
        # This torch version can reuse a cached graph after new hooks are added
        # without invoking them. Recompile explicitly, then gate its output.
        torch._dynamo.reset()
        try:
            compiled_capture, selection = route_predictions(model, samples, capture)
        finally:
            capture.close()
        del model, capture
        gc.collect()
        torch.cuda.empty_cache()
        model, mapping = load_eager_model(params, checkpoint, 0)
        eager_plain, _ = route_predictions(model, samples)
        observer = MechanismObserver(model)
        observer.variant = variant
        try:
            eager_observed, _ = route_predictions(model, samples, observer)
        finally:
            observer.close()
        del model, observer
        gc.collect()
        torch.cuda.empty_cache()
        summary = {
            "compiled_capture_vs_compiled_plain": comparison(compiled_capture, compiled_plain, atol, rtol),
            "eager_observer_vs_eager_plain": comparison(eager_observed, eager_plain, atol, rtol),
            "eager_plain_vs_compiled_plain": comparison(eager_plain, compiled_plain, atol, rtol),
            "eager_observer_vs_compiled_plain": comparison(eager_observed, compiled_plain, atol, rtol),
            "mapped_keys": mapping, "checkpoint_sha256": previous_gate["checkpoint_sha256"],
        }
        old = pd.read_parquet(args.failed_dir / f"{variant}_prediction_check.parquet")
        joined = meta.merge(old, on=list(meta.columns), how="left", validate="one_to_one", sort=False)
        old_compiled = {label: joined[f"{label}_compiled"].to_numpy() for label in LABELS}
        old_eager = {label: joined[f"{label}_eager"].to_numpy() for label in LABELS}
        summary["compiled_plain_vs_failed_run_compiled"] = comparison(compiled_plain, old_compiled, atol, rtol)
        summary["eager_observer_vs_failed_run_eager"] = comparison(eager_observed, old_eager, atol, rtol)
        atomic_json(output / f"{variant}_diagnosis.json", summary)
        pred = meta.copy()
        for name, values in (("compiled_plain", compiled_plain), ("compiled_capture", compiled_capture),
                             ("eager_plain", eager_plain), ("eager_observer", eager_observed)):
            for label in LABELS:
                pred[f"{label}_{name}"] = values[label]
        pred.to_parquet(output / f"{variant}_four_route_predictions.parquet", index=False)
        gate = summary["compiled_capture_vs_compiled_plain"]["passed"]
        table = pd.DataFrame(selection)
        if len(table) != 6 * len(meta) or table.duplicated(["part_id", "row_offset", "site"]).any():
            raise ValueError("Incomplete compiled selection capture")
        suffix = "validated" if gate else "unvalidated"
        table.to_parquet(output / f"{variant}_compiled_selection_{suffix}.parquet", index=False)
        summaries[variant] = {name: {key: value[key] for key in ("passed", "bitwise_equal")}
                              for name, value in summary.items() if isinstance(value, dict) and "passed" in value}
        if checkpoint_before != file_identity(checkpoint):
            raise ValueError("Checkpoint changed during diagnostic")
    atomic_json(output / "diagnosis_complete.json", {
        "variants": summaries, "original_tolerances_unchanged": True,
        "diagnostic_run_completed": True, "mechanism_analysis_complete": False,
        "am_pa_internal_states_not_recovered": True,
    })


if __name__ == "__main__":
    main()
