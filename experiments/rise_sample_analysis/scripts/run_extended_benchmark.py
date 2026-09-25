#!/usr/bin/env python3
"""Execute all six preregistered batch/order rounds without selecting winners."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter

from build_benchmark_cache import SIZES, verify_cache
from observe_mechanism import ROOT, atomic_json, parse_log, run_records, sha256, validate_parameters


def round_plan(cache_dir, native_log, output):
    rounds = []
    for index, size in enumerate(SIZES):
        for direction, order in (("forward", ["A", "B", "E"]), ("reverse", ["E", "B", "A"])):
            identifier = f"batch{size}_{direction}01"
            directory = output / identifier
            command = [sys.executable, str(ROOT / "experiments/rise_sample_analysis/scripts/benchmark_inference.py"),
                       "--sample-cache", str(cache_dir / "benchmark_batches.pt"),
                       "--sample-index", str(index), "--native-log", str(native_log),
                       "--mode", "compiled", "--variant-order", *order, "--round-id", identifier,
                       "--warmup", "1000", "--blocks", "10", "--repeats", "100",
                       "--output-dir", str(directory)]
            rounds.append({"round_id": identifier, "sample_index": index, "batch_rows": size,
                           "order": order, "output_dir": str(directory), "command": command})
    return rounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    source, cache, native, output = (value.resolve() for value in
                                    (args.source_dir, args.cache_dir, args.native_log, args.output_dir))
    if not output.is_relative_to(ROOT / "artifacts/rise_sample_analysis/mechanism"):
        parser.error("Output must be under artifacts/rise_sample_analysis/mechanism")
    cache_manifest = verify_cache(cache, source)
    benchmark_script = ROOT / "experiments/rise_sample_analysis/scripts/benchmark_inference.py"
    if sha256(benchmark_script) != cache_manifest["benchmark_script_sha256"]:
        raise ValueError("Frozen benchmark script changed since cache preparation")
    base, _ = parse_log(ROOT / run_records()["E"]["checkpoint_log"])
    native_params, _ = parse_log(native)
    validate_parameters(native_params, base, "A")
    if not native.with_suffix(".model").is_file():
        raise FileNotFoundError("Completed native A checkpoint is missing")
    rounds = round_plan(cache, native, output)
    if args.dry_run:
        print(json.dumps({"rounds": rounds, "total_variant_processes": 18,
                          "warmup_forwards_per_variant": 1000,
                          "measured_forwards_per_variant": 1000,
                          "native_protocol_passed": True, "cache_verified": True}, indent=2))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Extended GPU benchmark requires a Slurm allocation")
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    source_files = [Path(__file__).resolve(), Path(__file__).with_name("build_benchmark_cache.py"),
                    Path(__file__).with_name("submit_extended_benchmark.sbatch"), benchmark_script]
    manifest = {"started_utc": started_utc, "job_id": os.environ["SLURM_JOB_ID"],
                "cache_manifest_sha256": sha256(cache / "manifest.json"),
                "sample_cache_sha256": cache_manifest["benchmark_cache_sha256"],
                "native_log": str(native.relative_to(ROOT)), "native_log_sha256": sha256(native),
                "native_checkpoint": str(native.with_suffix(".model").relative_to(ROOT)),
                "native_checkpoint_sha256": sha256(native.with_suffix(".model")),
                "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_files},
                "rounds": rounds, "warmup": 1000, "blocks": 10, "repeats": 100,
                "A": "independently trained native baseline", "B": "content-only auxiliary scorer",
                "E": "full RISE", "select_best_round": False,
                "estimate_only": "30-minute debug allocation; actual wall time recorded separately",
                "comparison_scope": "same GPU resident real input batch within each size; full-model forward only"}
    atomic_json(output / "manifest.json", manifest)
    receipts, returncode = [], 1
    def terminated(signum, frame):
        nonlocal returncode
        returncode = 128 + signum
        raise SystemExit(returncode)
    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    try:
        for entry in rounds:
            round_start = perf_counter()
            atomic_json(output / "progress.json", {"active_round": entry["round_id"],
                        "completed_rounds": [row["round_id"] for row in receipts],
                        "elapsed_seconds": round_start - started})
            result = subprocess.run(entry["command"], check=False)
            receipt = {"round_id": entry["round_id"], "returncode": result.returncode,
                       "wall_seconds_including_load_compile_warmup_and_profile": perf_counter() - round_start,
                       "sample_index": entry["sample_index"], "batch_rows": entry["batch_rows"],
                       "order": entry["order"]}
            receipts.append(receipt)
            atomic_json(output / "round_receipts.json", receipts)
            if result.returncode:
                returncode = result.returncode
                raise RuntimeError(f"Round {entry['round_id']} failed with status {result.returncode}")
        all_records = []
        for entry in rounds:
            directory = Path(entry["output_dir"])
            complete = json.loads((directory / "complete.json").read_text())
            if complete["variants"] != entry["order"]:
                raise ValueError("Benchmark round variant order drift")
            for variant in entry["order"]:
                record = json.loads((directory / f"{variant}.json").read_text())
                if (record["batch_rows"] != entry["batch_rows"]
                        or record["sample_identity_sha256"] != cache_manifest["entries"][entry["sample_index"]]["identity_sha256"]
                        or record["warmup_forwards"] != 1000 or record["repeats_per_block"] != 100
                        or len(record["timing"]["block_mean_ms"]) != 10):
                    raise ValueError("Input identity or timing protocol drift")
                all_records.append(record)
        if len({row["environment_before"]["uuid"] for row in all_records}) != 1:
            raise ValueError("The six rounds did not run on one GPU")
        atomic_json(output / "complete.json", {"all_six_rounds_completed": True,
                    "variant_processes": len(all_records), "rounds": receipts,
                    "all_raw_block_measurements_retained": True, "no_best_round_selection": True,
                    "elapsed_seconds": perf_counter() - started})
        returncode = 0
    finally:
        atomic_json(output / "execution_outcome.json", {"returncode": returncode,
                    "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": perf_counter() - started, "completed_rounds": len(receipts),
                    "all_rounds_succeeded": returncode == 0,
                    "elapsed_scope": "orchestrator after preflight, including model load compile warmup timing and profile"})


if __name__ == "__main__":
    main()
