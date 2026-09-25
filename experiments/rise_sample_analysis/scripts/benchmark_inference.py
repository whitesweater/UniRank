#!/usr/bin/env python3
"""Warm single-GPU model-forward latency/memory for fixed B/E sample inputs.

Every variant runs in a fresh subprocess. Data loading, host-to-device transfer,
checkpoint loading and the first compiled forward are excluded from timings.
B is content-only, never the native baseline. Optional A requires its own log.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from observe_mechanism import (
    ROOT, LABELS, VARIANTS, atomic_json, device_metadata, file_identity, identity_hash,
    load_eager_model, load_model, load_sample_cache, parse_log, prediction_arrays,
    run_records, seed_everything, sha256, source_hashes, tree_map, validate_parameters,
)


def elapsed_summary(block_ms):
    values = np.asarray(block_ms, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Timing blocks must be finite positive values")
    return {"block_mean_ms": values.tolist(), "median_block_mean_ms": float(np.median(values)),
            "q25_block_mean_ms": float(np.quantile(values, 0.25)),
            "q75_block_mean_ms": float(np.quantile(values, 0.75)),
            "blocks": len(values), "tail_latency_not_measured": True}


def validate_variant_order(order, has_native):
    expected = {"B", "E", "A"} if has_native else {"B", "E"}
    order = list(order) if order is not None else ["B", "E"] + (["A"] if has_native else [])
    if len(order) != len(set(order)) or set(order) != expected:
        raise ValueError(f"Variant order must be a permutation of {sorted(expected)}")
    return order


def environment_snapshot():
    result = device_metadata()
    try:
        result["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,power.draw,clocks.sm,clocks.mem,memory.used",
             "--format=csv,noheader"], text=True).strip()
        result["compute_processes"] = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        result["nvidia_smi_error"] = str(error)
    return result


def worker(args):
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise RuntimeError("Benchmark requires one GPU process")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    records = run_records()
    base, _ = parse_log(ROOT / records["E"]["checkpoint_log"])
    variant = args.worker
    log = args.native_log.resolve() if variant == "A" else ROOT / records[variant]["checkpoint_log"]
    params, _ = parse_log(log)
    validate_parameters(params, base, variant)
    if variant != "A":
        params["sisa_variant"] = VARIANTS[variant]
    params.update(distributed=False, world_size=1, rank=0, local_rank=0)
    checkpoint = log.with_suffix(".model")
    checkpoint_before = file_identity(checkpoint)
    seed_everything(params["seed"])
    cache = load_sample_cache(args.sample_cache)
    if args.sample_index >= len(cache):
        raise ValueError("Sample index exceeds sample cache")
    sample = cache[args.sample_index]
    model = (load_model(params, checkpoint, 0) if args.mode == "compiled"
             else load_eager_model(params, checkpoint, 0)[0])
    if model.training:
        raise RuntimeError("Model unexpectedly in training mode")
    batch = tree_map(sample["batch"], lambda t: t.to("cuda:0"))
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    environment_before = environment_snapshot()
    torch.cuda.synchronize()
    first_start = perf_counter()
    with torch.no_grad(), model._get_amp_context():
        first = model.forward(batch)
    torch.cuda.synchronize()
    first_seconds = perf_counter() - first_start
    reference = prediction_arrays(first)
    if not all(np.isfinite(value).all() for value in reference.values()):
        raise ValueError("Nonfinite predictions during benchmark preflight")
    del first
    with torch.no_grad(), model._get_amp_context():
        for _ in range(args.warmup):
            result = model.forward(batch)
            del result
        torch.cuda.synchronize()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        block_ms = []
        for _ in range(args.blocks):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.repeats):
                result = model.forward(batch)
                del result
            end.record()
            end.synchronize()
            block_ms.append(start.elapsed_time(end) / args.repeats)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        final = prediction_arrays(model.forward(batch))
    for label in LABELS:
        if not np.array_equal(reference[label], final[label]):
            raise ValueError(f"Repeated eval predictions changed: {label}")
    # One profiling forward is outside both time and memory measurement windows.
    kernel_events = []
    profiler_error = None
    if not args.skip_profiler:
        try:
            with torch.no_grad(), model._get_amp_context(), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                model.forward(batch)
                torch.cuda.synchronize()
            needles = ("attention", "flash", "fmha", "softmax", "triton")
            kernel_events = sorted({event.name for event in prof.events()
                                    if any(token in event.name.lower() for token in needles)})
        except Exception as error:
            profiler_error = f"{type(error).__name__}: {error}"
    if checkpoint_before != file_identity(checkpoint):
        raise ValueError("Checkpoint changed during measurement")
    result = {
        "variant": variant, "name": "native_baseline" if variant == "A" else VARIANTS[variant],
        "round_id": args.round_id,
        "mode": args.mode, "timing_scope": "GPU-resident full-model forward, no loader or H2D; single GPU",
        "training": False, "no_grad": True, "bf16_autocast": bool(params["enable_bf16"]),
        "device_synchronized_at_boundaries": True, "fresh_variant_subprocess": True,
        "first_forward_seconds_excluded": first_seconds, "warmup_forwards": args.warmup,
        "repeats_per_block": args.repeats, "timing": elapsed_summary(block_ms),
        "batch_rows": len(sample["meta"]["row_offset"]), "max_history_length": int(batch[2].shape[1]),
        "sample_identity_sha256": identity_hash(sample["meta"]),
        "sample_cache_sha256": sha256(args.sample_cache), "sample_index": args.sample_index,
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
        "baseline_allocated_bytes": baseline_allocated, "peak_allocated_bytes": peak_allocated,
        "incremental_peak_allocated_bytes": peak_allocated - baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved, "peak_reserved_bytes": peak_reserved,
        "kernel_event_names": kernel_events, "profiler_error": profiler_error,
        "environment_before": environment_before, "environment_after": environment_snapshot(),
        "prediction_repeat_exact": True,
    }
    atomic_json(args.output_dir / f"{variant}.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-cache", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("compiled", "eager"), default="compiled")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--native-log", type=Path)
    parser.add_argument("--variant-order", nargs="+", choices=("A", "B", "E"),
                        help="Permutation of B E, or A B E when --native-log is provided")
    parser.add_argument("--round-id", default="round1", help="Label for this independent immutable output round")
    parser.add_argument("--skip-profiler", action="store_true")
    parser.add_argument("--worker", choices=("A", "B", "E"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.warmup, args.blocks, args.repeats) < 1 or args.sample_index < 0:
        parser.error("Positive timing repetitions and nonnegative sample index required")
    if not args.round_id.strip():
        parser.error("round-id must be nonempty")
    os.chdir(ROOT)
    args.output_dir = args.output_dir.resolve()
    args.sample_cache = args.sample_cache.resolve()
    if not args.output_dir.is_relative_to(ROOT / "artifacts/rise_sample_analysis/mechanism"):
        parser.error("Output must be below artifacts/rise_sample_analysis/mechanism")
    if args.worker:
        if args.worker == "A" and args.native_log is None:
            parser.error("Native A requires its completed checkpoint log")
        worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    for path in (Path(__file__).resolve(), Path(__file__).with_name("observe_mechanism.py")):
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    variants = validate_variant_order(args.variant_order, args.native_log is not None)
    manifest = {"variants_in_execution_order": variants, "mode": args.mode, "round_id": args.round_id,
                "source_sha256": hashes, "sample_cache_sha256": sha256(args.sample_cache),
                "warmup": args.warmup, "blocks": args.blocks, "repeats": args.repeats,
                "single_order_per_round": True,
                "latency_is_not_end_to_end": True, "no_training_performance_claim": True}
    atomic_json(args.output_dir / "manifest.json", manifest)
    for variant in variants:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", variant,
                   "--sample-cache", str(args.sample_cache), "--sample-index", str(args.sample_index),
                   "--output-dir", str(args.output_dir), "--mode", args.mode,
                   "--round-id", args.round_id,
                   "--warmup", str(args.warmup), "--blocks", str(args.blocks), "--repeats", str(args.repeats)]
        if args.native_log:
            command += ["--native-log", str(args.native_log.resolve())]
        if args.skip_profiler:
            command += ["--skip-profiler"]
        subprocess.run(command, check=True)
    records = {variant: json.loads((args.output_dir / f"{variant}.json").read_text()) for variant in variants}
    if len({item["sample_identity_sha256"] for item in records.values()}) != 1:
        raise ValueError("Benchmark variants used different inputs")
    for key in ("name", "uuid", "torch", "cuda_build", "node", "cuda_visible_devices"):
        if len({str(item["environment_before"][key]) for item in records.values()}) != 1:
            raise ValueError(f"Benchmark environment differs for {key}")
    times = {variant: item["timing"]["median_block_mean_ms"] for variant, item in records.items()}
    atomic_json(args.output_dir / "complete.json", {
        "variants": variants, "round_id": args.round_id, "median_model_forward_ms": times,
        "E_over_B_content_only": times["E"] / times["B"],
        "E_over_A_native": times["E"] / times["A"] if "A" in times else None,
        "same_sample_and_environment": True, "single_gpu_inference_only": True,
        "kernel_path_inspection_required": True,
    })


if __name__ == "__main__":
    main()
