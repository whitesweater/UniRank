#!/usr/bin/env python3
"""Observe fixed-representation HyFormer--MerRec B/E selection mechanisms.

No training or core patching. Eager hooks are accepted only after rowwise
prediction comparison against the same compiled checkpoint. Native padding
semantics are preserved; reconstructed softmax weights are not fused-kernel
attention outputs. No attention matrix is written to disk.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from export_predictions import (
    ROOT, LABELS, VARIANTS, atomic_json, build_loader, canonical_hash, file_identity,
    identity_hash, load_model, parse_log, run_records, sha256, source_hashes,
    validate_parameters, verify_shard,
)
from unirank.features import feature_map_from_config
from unirank.pytorch.layers.attentions.sisa import (
    SISAScoreBias, _deterministic_prefix_sum, _reshape_heads, _rotate_pairs,
)
from unirank.pytorch.torch_utils import seed_everything


def eager_checkpoint_state(state, expected):
    """Remove only complete torch.compile wrapper path segments; reject aliases."""
    mapped = {}
    mapping = {}
    for source, tensor in state.items():
        target = ".".join(part for part in source.split(".") if part != "_orig_mod")
        if target in mapped:
            raise ValueError(f"Checkpoint key collision after wrapper mapping: {target}")
        if (target not in expected or tensor.shape != expected[target].shape
                or tensor.dtype != expected[target].dtype):
            raise ValueError(f"Unexpected key/shape after wrapper mapping: {source}")
        mapped[target], mapping[source] = tensor, target
    if set(mapped) != set(expected):
        raise ValueError(f"Missing eager checkpoint keys: {sorted(set(expected) - set(mapped))}")
    return mapped, mapping


def load_eager_model(params, checkpoint, device):
    from model_zoo.HyFormer import HyFormer
    configured = dict(params, gpu=device, enable_torch_compile=False)
    model = HyFormer(feature_map_from_config(configured), **configured)
    if model._torch_compile_enabled:
        raise RuntimeError("Observation model unexpectedly compiled")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    mapped, mapping = eager_checkpoint_state(state, model.state_dict())
    model.load_state_dict(mapped, strict=True)
    del state, mapped
    model.optimizers = []
    model.optimizer = model.dense_optimizer = model.sparse_optimizer = None
    model.eval()
    return model, mapping


def tree_map(value, function):
    if torch.is_tensor(value):
        return function(value)
    if isinstance(value, dict):
        return {key: tree_map(item, function) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tree_map(item, function) for item in value)
    raise TypeError(f"Unsupported batch node {type(value)}")


def subset_batch(batch, indices):
    indices = torch.as_tensor(indices, dtype=torch.long)
    size = batch[2].shape[0]
    def select(tensor):
        if tensor.shape[0] != size:
            raise ValueError("Every collated tensor must have a leading example dimension")
        return tensor.index_select(0, indices.to(tensor.device)).clone()
    return tree_map(batch, select)


def deterministic_indices(meta, count, seed):
    """Uniform hash priorities within a declared loader window; no label inputs."""
    priorities = [hashlib.sha256(f"{seed}:{int(p)}:{int(r)}".encode()).digest()
                  for p, r in zip(meta["part_id"], meta["row_offset"])]
    return np.asarray(sorted(sorted(range(len(priorities)), key=priorities.__getitem__)[:count]),
                      dtype=np.int64)


def collect_samples(params, *, batches_per_rank, samples_per_batch, seed, num_workers):
    samples, frame = [], []
    seen = set()
    for rank in (0, 1):
        _, loader = build_loader(params, rank, 2, num_workers)
        for batch_number, (batch, meta) in enumerate(loader):
            if batch_number >= batches_per_rank:
                break
            indices = deterministic_indices(meta, samples_per_batch, seed)
            selected = {key: values[indices].copy() for key, values in meta.items()}
            ids = list(zip(selected["part_id"].tolist(), selected["row_offset"].tolist()))
            if seen.intersection(ids):
                raise ValueError("Duplicate selected source row")
            seen.update(ids)
            samples.append({"batch": subset_batch(batch, indices), "meta": selected})
            frame.append({"source_rank": rank, "loader_batch": batch_number,
                          "part_id": int(meta["part_id"][0]),
                          "original_row_start": int(meta["row_offset"][0]),
                          "original_rows": len(meta["row_offset"]), "selected_rows": len(indices),
                          "selected_identity_sha256": identity_hash(selected)})
        del loader
        gc.collect()
    if not samples:
        raise ValueError("Sampling frame is empty")
    return samples, frame


def save_sample_cache(path, samples):
    serializable = [{"batch": item["batch"],
                     "meta": {key: torch.from_numpy(value) for key, value in item["meta"].items()}}
                    for item in samples]
    torch.save(serializable, path)


def load_sample_cache(path):
    items = torch.load(path, map_location="cpu", weights_only=True)
    return [{"batch": item["batch"],
             "meta": {key: value.numpy() for key, value in item["meta"].items()}}
            for item in items]


def native_allowed(query, key, mask=None, is_causal=False):
    shape = (*query.shape[:-1], key.shape[-2])
    allowed = torch.ones(shape, dtype=torch.bool, device=query.device)
    if mask is not None:
        if mask.dtype != torch.bool:
            raise ValueError("Only native boolean hard masks are supported by this observer")
        allowed &= mask.expand(shape)
    if is_causal:
        allowed &= torch.ones(shape[-2:], dtype=torch.bool, device=query.device).tril()
    return allowed


def masked_mean(values, mask):
    mask = mask.expand_as(values)
    axes = tuple(range(1, values.ndim))
    count = mask.sum(dim=axes)
    total = values.masked_fill(~mask, 0.0).sum(dim=axes)
    return torch.where(count > 0, total / count.clamp_min(1), torch.full_like(total, torch.nan))


def selection_statistics(query, key, delta, *, scale=None, mask=None, is_causal=False,
                         query_valid=None, key_valid=None):
    """Per-example exact mathematical softmax summaries in FP32, before dropout."""
    # Disable enclosing BF16 autocast for the observational reconstruction.
    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
        scores = scores / (float(scale) if scale is not None else query.shape[-1] ** 0.5)
        delta = torch.zeros_like(scores) if delta is None else delta.float().expand_as(scores)
        allowed = native_allowed(query, key, mask, is_causal)
        any_allowed = allowed.any(dim=-1)
        row_valid = any_allowed.clone()
        if query_valid is not None:
            row_valid &= query_valid[:, None, :]
        safe_scores = scores.masked_fill(~allowed, -torch.inf)
        safe_corrected = (scores + delta).masked_fill(~allowed, -torch.inf)
        # Avoid NaNs for fully masked queries; exclude these rows from all means.
        safe_scores = torch.where(any_allowed[..., None], safe_scores, torch.zeros_like(safe_scores))
        safe_corrected = torch.where(any_allowed[..., None], safe_corrected, torch.zeros_like(safe_corrected))
        native = torch.softmax(safe_scores, dim=-1)
        corrected = torch.softmax(safe_corrected, dim=-1)
        tv = 0.5 * (native - corrected).abs().sum(dim=-1)
        maximum = delta.masked_fill(~allowed, -torch.inf).amax(dim=-1)
        minimum = delta.masked_fill(~allowed, torch.inf).amin(dim=-1)
        omega = torch.where(any_allowed, maximum - minimum, torch.zeros_like(maximum))
        bound = torch.tanh(omega / 4)
        changed = (safe_scores.argmax(-1) != safe_corrected.argmax(-1)).float()
        pair_valid = allowed & row_valid[..., None]
        key_count = allowed.sum(-1).clamp_min(1)
        native_center = scores - scores.masked_fill(~allowed, 0).sum(-1)[..., None] / key_count[..., None]
        delta_center = delta - delta.masked_fill(~allowed, 0).sum(-1)[..., None] / key_count[..., None]
        score_sd = torch.sqrt((native_center.square().masked_fill(~allowed, 0).sum(-1) / key_count).clamp_min(0))
        delta_sd = torch.sqrt((delta_center.square().masked_fill(~allowed, 0).sum(-1) / key_count).clamp_min(0))
        result = {
            "valid_attention_rows": row_valid.sum((1, 2)),
            "allowed_pairs": pair_valid.sum((1, 2, 3)),
            "omega_mean": masked_mean(omega, row_valid),
            "omega_max": omega.masked_fill(~row_valid, 0).amax((1, 2)),
            "tv_mean": masked_mean(tv, row_valid),
            "tv_max": tv.masked_fill(~row_valid, 0).amax((1, 2)),
            "tv_bound_mean": masked_mean(bound, row_valid),
            "tv_bound_violation_max": (tv - bound).masked_fill(~row_valid, 0).amax((1, 2)),
            "top_key_changed_fraction": masked_mean(changed, row_valid),
            "delta_mean": masked_mean(delta, pair_valid),
            "delta_abs_mean": masked_mean(delta.abs(), pair_valid),
            "delta_positive_fraction": masked_mean((delta > 0).float(), pair_valid),
            "delta_negative_fraction": masked_mean((delta < 0).float(), pair_valid),
            "native_logit_sd_mean": masked_mean(score_sd, row_valid),
            "delta_centered_sd_mean": masked_mean(delta_sd, row_valid),
            "delta_native_sd_ratio_mean": masked_mean(delta_sd / score_sd.clamp_min(1e-12), row_valid),
        }
        if scores.shape[-1] >= 2:
            top_scores = safe_scores.topk(2, dim=-1).values
            gap = top_scores[..., 0] - top_scores[..., 1]
            comparable = row_valid & (allowed.sum(-1) >= 2)
            certified = gap > omega
            result["native_top_gap_mean"] = masked_mean(gap, comparable)
            result["native_top_order_certified_fraction"] = masked_mean(certified.float(), comparable)
            result["top_order_certificate_violation_fraction"] = masked_mean(
                (certified & changed.bool()).float(), comparable)
        if key_valid is not None:
            padding = ~key_valid[:, None, None, :]
            result["native_padding_mass_mean"] = masked_mean((native * padding).sum(-1), row_valid)
            result["corrected_padding_mass_mean"] = masked_mean((corrected * padding).sum(-1), row_valid)
            content_pairs = pair_valid & ~padding
            result["delta_abs_mean_valid_keys"] = masked_mean(delta.abs(), content_pairs)
            result["delta_positive_fraction_valid_keys"] = masked_mean((delta > 0).float(), content_pairs)
            result["delta_negative_fraction_valid_keys"] = masked_mean((delta < 0).float(), content_pairs)
        return result


def control_statistics(module, hidden, valid_mask, head_dim, query_slice=None, key_slice=None):
    """Auxiliary algebra at fixed hidden vectors; no retraining intervention."""
    valid_mask = valid_mask.bool()
    raw_b = _reshape_heads(F.linear(hidden, module.b_weight), module.num_heads, module.score_dim)
    raw_c = _reshape_heads(F.linear(hidden, module.c_weight), module.num_heads, module.score_dim)
    if module.use_rotation:
        phase = _reshape_heads(F.linear(hidden, module.phase_weight), module.num_heads,
                               module.score_dim // 2).float()
    else:
        phase = torch.zeros_like(raw_b[..., :module.score_dim // 2], dtype=torch.float32)
    if module.use_decay:
        log_alpha = -F.softplus(F.linear(hidden.float(), module.decay_weight.float(),
                                       module.decay_projection_bias.float())).permute(0, 2, 1)
    else:
        log_alpha = torch.zeros_like(raw_b[..., 0], dtype=torch.float32)
    phase = phase.masked_fill(~valid_mask[:, None, :, None], 0)
    log_alpha = log_alpha.masked_fill(~valid_mask[:, None, :], 0)
    decay = _deterministic_prefix_sum(log_alpha, dim=2)
    cumulative_phase = _deterministic_prefix_sum(phase, dim=2)
    offset = module._minimax_offset(decay, valid_mask)
    if module.decay_reference == "query":
        query_decay = decay
    else:
        endpoint = decay.masked_fill(~valid_mask[:, None, :], torch.inf).amin(-1)
        endpoint = torch.where(valid_mask.any(-1)[:, None], endpoint, torch.zeros_like(endpoint))
        query_decay = endpoint[..., None].expand_as(decay)
    qpre, kpre = query_decay - offset[..., None], decay - offset[..., None]
    qslice, kslice = query_slice or slice(None), key_slice or slice(None)
    qvalid, kvalid = valid_mask[:, qslice], valid_mask[:, kslice]
    pair_valid = qvalid[:, None, :, None] & kvalid[:, None, None, :]
    token_valid = valid_mask[:, None, :]
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        exponent = (qpre[:, :, qslice].clamp(-module.exp_clamp, module.exp_clamp)[..., None]
                    - kpre[:, :, kslice].clamp(-module.exp_clamp, module.exp_clamp)[:, :, None, :])
        c, b = raw_c[:, :, qslice].float(), raw_b[:, :, kslice].float()
        rc = _rotate_pairs(c, cumulative_phase[:, :, qslice])
        rb = _rotate_pairs(b, cumulative_phase[:, :, kslice])
        normalizer = c.norm(dim=-1)[..., None] * b.norm(dim=-1)[:, :, None, :]
        content = torch.matmul(c, b.transpose(-1, -2))
        rotated = torch.matmul(rc, rb.transpose(-1, -2))
        cosine_change = (rotated - content) / normalizer.clamp_min(1e-12)
        lambdas = module.lambdas(dtype=torch.float32, device=hidden.device)[None, :, None, None]
        amplitude = exponent.exp()
        result = {
            "am_log_amplitude_mean_valid_pairs": masked_mean(exponent, pair_valid),
            "am_amplitude_mean_valid_pairs": masked_mean(amplitude, pair_valid),
            "am_pair_amplification_fraction": masked_mean((exponent > 0).float(), pair_valid),
            "log_decay_increment_mean": masked_mean(log_alpha, token_valid),
            "phase_increment_abs_mean": masked_mean(phase.abs(), token_valid[..., None]),
            "phase_prefix_abs_mean": masked_mean(cumulative_phase.abs(), token_valid[..., None]),
            "query_clipping_fraction": masked_mean((qpre[:, :, qslice].abs() > module.exp_clamp).float(),
                                                     qvalid[:, None, :]),
            "key_clipping_fraction": masked_mean((kpre[:, :, kslice].abs() > module.exp_clamp).float(),
                                                   kvalid[:, None, :]),
            "pa_cosine_change_abs_mean_valid_pairs": masked_mean(cosine_change.abs(), pair_valid),
            "pa_alignment_sign_change_fraction": masked_mean(((content * rotated) < 0).float(), pair_valid),
            "frozen_pa_delta_effect_abs_mean": masked_mean(
                (lambdas * amplitude * (rotated - content) * module.score_scale).abs(), pair_valid),
            "frozen_am_delta_effect_abs_mean": masked_mean(
                (lambdas * (amplitude - 1) * rotated * module.score_scale).abs(), pair_valid),
        }
    return result, qvalid, kvalid


class MechanismObserver:
    def __init__(self, model, expected_sites=6):
        if model.training:
            raise ValueError("Observation requires model.eval()")
        self.handles, self.pending, self.rows = [], {}, []
        self.meta = None
        self.variant = None
        self.site_names = []
        for name, parent in model.named_modules():
            bias = getattr(parent, "sisa_score_bias", None)
            attention = getattr(parent, "dot_attention", None)
            if not isinstance(bias, SISAScoreBias) or attention is None:
                continue
            if attention.attention_activation_type != "SoftMax":
                raise ValueError("TV observer supports softmax sites only")
            self.site_names.append(name)
            self.handles.append(bias.register_forward_pre_hook(self._bias_hook(name), with_kwargs=True))
            self.handles.append(attention.register_forward_pre_hook(self._attention_hook(name, bias),
                                                                    with_kwargs=True))
        if len(self.site_names) != expected_sites:
            self.close()
            raise ValueError(f"Expected {expected_sites} attention sites, found {self.site_names}")

    def _bias_hook(self, site):
        def hook(module, args, kwargs):
            if site in self.pending:
                raise RuntimeError("Auxiliary capture was not consumed")
            self.pending[site] = (args, kwargs)
        return hook

    def _attention_hook(self, site, bias):
        def hook(module, args, kwargs):
            if self.meta is None or site not in self.pending:
                raise RuntimeError("Observer requires metadata and a matching auxiliary forward")
            aux_args, aux_kwargs = self.pending.pop(site)
            control, query_valid, key_valid = control_statistics(bias, *aux_args, **aux_kwargs)
            selection = selection_statistics(args[0], args[1], kwargs.get("score_bias"),
                scale=kwargs.get("scale"), mask=kwargs.get("mask"),
                is_causal=kwargs.get("is_causal", False), query_valid=query_valid, key_valid=key_valid)
            arrays = {key: value.detach().float().cpu().numpy() for key, value in (selection | control).items()}
            lambda_mean = float(bias.lambdas(dtype=torch.float32, device=args[0].device).mean())
            for index in range(len(self.meta["row_offset"])):
                row = {key: int(value[index]) for key, value in self.meta.items()}
                row.update(variant=self.variant, site=site, decay_reference=bias.decay_reference,
                           heads=bias.num_heads, score_dim=bias.score_dim,
                           native_mask_present=kwargs.get("mask") is not None,
                           lambda_mean=lambda_mean)
                row.update({key: float(value[index]) for key, value in arrays.items()})
                self.rows.append(row)
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()


def prediction_arrays(output):
    return {label: output[f"{label}_pred"].detach().float().cpu().numpy().reshape(-1) for label in LABELS}


def prediction_gate(actual, reference, atol, rtol):
    details = {}
    for label in LABELS:
        left, right = np.asarray(actual[label]), np.asarray(reference[label])
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"Prediction shape/finite gate failed: {label}")
        absolute = np.abs(left - right)
        passed = absolute <= atol + rtol * np.abs(right)
        details[label] = {"rows": int(left.size), "max_abs": float(absolute.max(initial=0)),
                          "mean_abs": float(absolute.mean()), "failed_rows": int((~passed).sum()),
                          "passed": bool(passed.all())}
    return {"atol": atol, "rtol": rtol, "labels": details,
            "passed": all(item["passed"] for item in details.values())}


def exported_predictions(directory, variant, samples, checkpoint_hash):
    manifest = json.loads((directory / variant / "manifest.json").read_text())
    if manifest["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Exported predictions use a different checkpoint")
    shared = json.loads((directory / "manifest.json").read_text())
    if not shared.get("compile") or not shared.get("bf16"):
        raise ValueError("Export reference must be compiled BF16")
    size = shared["batch_size"]
    loaded, outputs = {}, {label: [] for label in LABELS}
    for sample in samples:
        meta = sample["meta"]
        for i, (part, row) in enumerate(zip(meta["part_id"], meta["row_offset"])):
            start = int(row) // size * size
            path = directory / variant / f"part-{int(part):05d}-offset-{start:012d}.parquet"
            if path not in loaded:
                verify_shard(path)
                loaded[path] = pq.read_table(path).to_pandas().set_index("row_offset", verify_integrity=True)
            source = loaded[path].loc[int(row)]
            for column, values in meta.items():
                if int(source[column]) != int(values[i]):
                    raise ValueError(f"Export row identity mismatch for {column}")
            for label in LABELS:
                outputs[label].append(float(source[f"{label}_pred"]))
    return {key: np.asarray(value, dtype=np.float32) for key, value in outputs.items()}


def device_metadata():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {"name": prop.name, "total_memory": prop.total_memory,
            "uuid": str(getattr(prop, "uuid", "not_exposed")),
            "capability": [prop.major, prop.minor], "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "node": os.uname().nodename,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=("B", "E"), default=["B", "E"])
    parser.add_argument("--batches-per-rank", type=int, default=2)
    parser.add_argument("--samples-per-batch", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--sample-cache", type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--prediction-atol", type=float, default=5e-4)
    parser.add_argument("--prediction-rtol", type=float, default=5e-3)
    args = parser.parse_args()
    if min(args.batches_per_rank, args.samples_per_batch) < 1 or args.num_workers < 0:
        parser.error("Positive sample sizes and nonnegative workers required")
    if min(args.prediction_atol, args.prediction_rtol) < 0 or len(set(args.variants)) != len(args.variants):
        parser.error("Invalid tolerance or duplicate variants")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1 or not torch.cuda.is_available():
        raise RuntimeError("Observer requires one GPU process; this is inference, not a training run")
    os.chdir(ROOT)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "artifacts/rise_sample_analysis/mechanism"):
        parser.error("Output must be under artifacts/rise_sample_analysis/mechanism")
    output.mkdir(parents=True, exist_ok=False)
    records = run_records()
    base, _ = parse_log(ROOT / records["E"]["checkpoint_log"])
    if args.sample_cache:
        samples, frame = load_sample_cache(args.sample_cache), None
    else:
        samples, frame = collect_samples(base, batches_per_rank=args.batches_per_rank,
            samples_per_batch=args.samples_per_batch, seed=args.sample_seed, num_workers=args.num_workers)
    save_sample_cache(output / "sample_batches.pt", samples)
    sample_table = pd.concat([pd.DataFrame(item["meta"]) for item in samples], ignore_index=True)
    if sample_table.duplicated(["part_id", "row_offset"]).any():
        raise ValueError("Duplicate cached source rows")
    sample_table.to_parquet(output / "sample_rows.parquet", index=False)
    hashes = source_hashes()
    hashes[str(Path(__file__).resolve().relative_to(ROOT))] = sha256(Path(__file__))
    manifest = {"device": device_metadata(), "variants": args.variants, "rows": len(sample_table),
                "source_sha256": hashes, "sample_cache_sha256": sha256(output / "sample_batches.pt"),
                "sample_frame": frame, "sample_seed": args.sample_seed,
                "sampling_scope": "hash-sampled rows within first declared batches of both historical ranks; not population-representative",
                "loaded_sample_cache": str(args.sample_cache) if args.sample_cache else None,
                "prediction_atol": args.prediction_atol, "prediction_rtol": args.prediction_rtol,
                "fixed_representation_selection": True, "attention_reconstruction": "FP32 pre-dropout mathematical softmax from actual Q/K and delta",
                "padding_rule": "native allowed keys retained; invalid history queries excluded from row averages",
                "no_attention_matrices_persisted": True, "bf16": True, "training": False,
                "grad_enabled": False, "compiled_reference": True, "eager_observer": True}
    atomic_json(output / "manifest.json", manifest)
    gates = {}
    for variant in args.variants:
        log = ROOT / records[variant]["checkpoint_log"]
        checkpoint = log.with_suffix(".model")
        params, _ = parse_log(log)
        validate_parameters(params, base, variant)
        params.update(sisa_variant=VARIANTS[variant], distributed=False, world_size=1, rank=0, local_rank=0)
        checkpoint_hash = sha256(checkpoint)
        before = file_identity(checkpoint)
        seed_everything(params["seed"])
        model = load_model(params, checkpoint, 0)
        compiled = {label: [] for label in LABELS}
        started = perf_counter()
        for item in samples:
            with torch.no_grad(), model._get_amp_context():
                values = prediction_arrays(model.forward(tree_map(item["batch"], lambda t: t.clone())))
            for label in LABELS:
                compiled[label].append(values[label])
        compiled = {label: np.concatenate(values) for label, values in compiled.items()}
        del model
        gc.collect()
        torch.cuda.empty_cache()
        compiled_wall = perf_counter() - started
        model, mapping = load_eager_model(params, checkpoint, 0)
        observer = MechanismObserver(model)
        eager = {label: [] for label in LABELS}
        try:
            for item in samples:
                observer.meta, observer.variant = item["meta"], variant
                with torch.no_grad(), model._get_amp_context():
                    values = prediction_arrays(model.forward(tree_map(item["batch"], lambda t: t.clone())))
                for label in LABELS:
                    eager[label].append(values[label])
            if observer.pending:
                raise RuntimeError("Unconsumed site captures")
        finally:
            observer.close()
        table = pd.DataFrame(observer.rows)
        expected_rows = len(sample_table) * 6
        if len(table) != expected_rows or table.duplicated(["part_id", "row_offset", "site"]).any():
            raise ValueError("Incomplete/duplicate site observations")
        table.to_parquet(output / f"{variant}_observations_unvalidated.parquet", index=False)
        eager = {label: np.concatenate(values) for label, values in eager.items()}
        gate = prediction_gate(eager, compiled, args.prediction_atol, args.prediction_rtol)
        gate.update(variant=variant, checkpoint_sha256=checkpoint_hash, checkpoint_file=before,
                    mapped_checkpoint_keys=mapping, compiled_reference_wall_seconds=compiled_wall,
                    compile_time_included_in_reference_wall=True)
        if args.prediction_dir:
            exported = exported_predictions(args.prediction_dir.resolve(), variant, samples, checkpoint_hash)
            gate["eager_vs_export"] = prediction_gate(eager, exported, args.prediction_atol, args.prediction_rtol)
            gate["compiled_microbatch_vs_export"] = prediction_gate(compiled, exported, args.prediction_atol, args.prediction_rtol)
            gate["passed"] &= gate["eager_vs_export"]["passed"] and gate["compiled_microbatch_vs_export"]["passed"]
        if before != file_identity(checkpoint):
            raise ValueError("Checkpoint changed during observation")
        atomic_json(output / f"{variant}_prediction_gate.json", gate)
        pred_table = sample_table.copy()
        for label in LABELS:
            pred_table[f"{label}_eager"] = eager[label]
            pred_table[f"{label}_compiled"] = compiled[label]
        pred_table.to_parquet(output / f"{variant}_prediction_check.parquet", index=False)
        gates[variant] = gate["passed"]
        if gate["passed"]:
            (output / f"{variant}_observations_unvalidated.parquet").rename(output / f"{variant}_observations.parquet")
            numeric = table.select_dtypes(include=["number"]).columns.difference(
                list(sample_table.columns) + ["heads", "score_dim"])
            summary = table.groupby("site", sort=True)[numeric].mean()
            summary.to_csv(output / f"{variant}_site_means.csv")
        del model, observer, table
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(output / "validation.json", {"all_passed": all(gates.values()), "variants": gates})
    if not all(gates.values()):
        raise RuntimeError("Numerical gate failed: unvalidated observations must not support mechanism claims")
    atomic_json(output / "complete.json", {"rows": len(sample_table), "sites_per_variant": 6,
                "variants": args.variants, "prediction_gates_passed": True,
                "observational_only": True, "manifest_sha256": canonical_hash(manifest)})


if __name__ == "__main__":
    main()
