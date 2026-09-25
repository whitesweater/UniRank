"""One-GPU inference over both historical logical ranks, without training.

The model runs in eval mode on one physical process. The data loader retains
world_size=2, rank-specific block assignment, batch 16384 and per-block drop_last.
This is serial execution of the original evaluation batches, not a single-rank
reinterpretation of the evaluation split. All metrics pool both logical ranks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import json
import logging
import os
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import export_predictions as engine
import torch
import torch.distributed as dist


ROOT = engine.ROOT
LOGICAL_WORLD_SIZE = 2


def logical_coverage(dataset) -> list[dict[int, int]]:
    buckets, _ = dataset._assign_blocks_greedily(dataset.blocks, LOGICAL_WORLD_SIZE)
    return [{int(block["part_id"]): int(block["num_rows"]) // dataset.batch_size * dataset.batch_size
             for block in bucket} for bucket in buckets]


def merge_coverage(allocation: list[dict[int, int]]) -> dict[int, int]:
    if len(allocation) != LOGICAL_WORLD_SIZE:
        raise ValueError("Exactly two historical logical ranks are required")
    expected = {}
    for per_rank in allocation:
        if set(expected) & set(per_rank):
            raise ValueError("Logical ranks overlap in source part allocation")
        expected.update(per_rank)
    return expected


def loader_for_rank(params: dict, logical_rank: int, workers: int):
    if logical_rank not in range(LOGICAL_WORLD_SIZE):
        raise ValueError("Unexpected logical rank")
    return engine.build_loader(params, logical_rank, LOGICAL_WORLD_SIZE, workers)


def assert_eval_invariants(model) -> dict:
    if any(module.training for module in model.modules()):
        raise ValueError("Every module must be in evaluation mode")
    if model.distributed or model._ddp_model is not None or dist.is_initialized():
        raise ValueError("Serial inference must not use an initialized distributed model/group")
    if any(isinstance(module, torch.nn.SyncBatchNorm) for module in model.modules()):
        raise ValueError("Unexpected synchronization module in frozen HyFormer")
    return {"all_modules_eval": True, "distributed_model": False,
            "distributed_process_group_initialized": False, "training_executed": False}


def source_hashes() -> dict:
    hashes = engine.source_hashes()
    hashes[str(Path(__file__).relative_to(ROOT))] = engine.sha256(Path(__file__))
    return hashes


def shutdown_loader(loader) -> None:
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()


def serial_variant(params: dict, checkpoint: Path, directory: Path, allocation: list[dict[int, int]],
                   data_version: str, workers: int, limit_batches: int | None, resume: bool) -> dict:
    """Load once, then execute original rank 0 and rank 1 batches in sequence."""
    configured = dict(params, distributed=False, rank=0, local_rank=0, world_size=1, gpu=0)
    engine.seed_everything(params["seed"])
    model = None
    invariants = None
    try:
        for logical_rank in range(LOGICAL_WORLD_SIZE):
            dataset, loader = loader_for_rank(params, logical_rank, workers)
            if dataset.num_batches * params["batch_size"] != sum(allocation[logical_rank].values()):
                raise ValueError(f"Logical rank {logical_rank} coverage drift")
            receipts = []
            try:
                for index, (batch, meta) in enumerate(loader):
                    if limit_batches is not None and index >= limit_batches:
                        break
                    path = directory / engine.shard_name(meta)
                    if resume and path.exists():
                        if path.with_suffix(".json").exists():
                            receipts.append(engine.verify_shard(path, engine.identity_hash(meta)))
                            continue
                        os.replace(path, path.with_name(path.name + f".uncommitted-{os.getpid()}"))
                    if model is None:
                        model = engine.load_model(configured, checkpoint, 0)
                        invariants = assert_eval_invariants(model)
                    with torch.no_grad(), model._get_amp_context():
                        predictions = model.forward(batch)
                    values = {label: predictions[f"{label}_pred"].float().detach().cpu().numpy().reshape(-1)
                              for label in engine.LABELS}
                    receipts.append(engine.write_shard(directory, meta, values))
                    if index % 20 == 0:
                        logging.info("logical_rank=%d batch=%d physical_gpu=0", logical_rank, index)
                rows = engine.validate_receipts(receipts, allocation[logical_rank], params["batch_size"],
                                                limit_batches is not None)
                engine.atomic_json(directory / f"rank-{logical_rank}.json", {
                    "rank": logical_rank, "logical_rank": logical_rank, "physical_rank": 0,
                    "physical_gpu": 0, "shards": receipts, "rows": rows, "data_version": data_version})
            finally:
                shutdown_loader(loader)
                del loader, dataset
                gc.collect()
        return invariants or {"resume_reused_existing_predictions": True, "model_loaded_in_this_attempt": False}
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(engine.VARIANTS), default=list(engine.COMPONENTS))
    parser.add_argument("--extra-baseline-log", type=Path)
    parser.add_argument("--baseline-task-id", default="not_recorded")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-batches", type=int, help="Smoke only: limit for EACH logical rank and variant")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--metric-atol", type=float, default=1e-6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit_batches is not None and args.limit_batches < 1:
        parser.error("--limit-batches must be positive")
    if args.num_workers < 0 or args.metric_atol < 0 or len(set(args.variants)) != len(args.variants):
        parser.error("Invalid worker count, tolerance, or duplicate variants")
    if ("A" in args.variants) != (args.extra_baseline_log is not None):
        parser.error("A and --extra-baseline-log must be supplied together")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or dist.is_initialized():
        parser.error("Run one Python process, without torchrun/distributed initialization")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Serial export requires exactly one visible physical GPU")
    if engine.DATASET != "MerRec_Action" or tuple(engine.LABELS) != ("Like", "Cart", "Offer", "Checkout", "Purchase"):
        raise RuntimeError("This serial exporter is restricted to HyFormer--MerRec")
    os.chdir(ROOT)
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "artifacts/rise_sample_analysis"):
        parser.error("Output must be below artifacts/rise_sample_analysis")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    output.mkdir(parents=True, exist_ok=True)
    output_lock = (output / ".export.lock").open("a")
    fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records = engine.run_records()
    if args.extra_baseline_log is not None:
        extra = engine.validate_extra_log(args.extra_baseline_log)
        records["A"] = {"checkpoint_log": str(extra.relative_to(ROOT)), "job_task_id": args.baseline_task_id}
    base_params, _ = engine.parse_log(ROOT / records["E"]["checkpoint_log"])
    required = {"model": "HyFormer", "dataset_id": "MerRec_Action", "batch_size": 16384,
                "world_size": 2, "seed": 20262028, "max_len": 100,
                "enable_bf16": True, "enable_torch_compile": True}
    if any(base_params.get(key) != value for key, value in required.items()):
        raise ValueError("Unexpected historical protocol")
    dataset, planning_loader = loader_for_rank(base_params, 0, 0)
    allocation = logical_coverage(dataset)
    expected = merge_coverage(allocation)
    if sum(expected.values()) != engine.FORMAL_ROWS:
        raise ValueError("Historical full sample coverage does not match")
    files = sorted({str(block[key]) for block in dataset.blocks
                    for key in ("data_file", "user_info_file", "item_info_file")}
                   | {str(engine.find_meta_data_json(base_params["test_user_info"]))})
    del dataset, planning_loader
    shared = {"schema_version": engine.SCHEMA_VERSION, "dataset": engine.DATASET, "split": "test",
              "model": "HyFormer", "evaluation_execution": "serial_historical_logical_ranks",
              "physical_gpus": 1, "physical_processes": 1, "world_size": 1, "logical_world_size": LOGICAL_WORLD_SIZE,
              "batch_size": base_params["batch_size"], "limit_batches_per_logical_rank": args.limit_batches,
              "num_workers": args.num_workers, "metric_atol": args.metric_atol,
              "data_files": [engine.file_identity(Path(path)) for path in files],
              "source_sha256": source_hashes(), "torch_version": torch.__version__,
              "cuda_version": torch.version.cuda, "physical_gpu_name": torch.cuda.get_device_name(0),
              "expected_rows_by_part": expected, "expected_rows_by_logical_rank": [sum(parts.values()) for parts in allocation],
              "expected_parts_by_logical_rank": allocation, "labels": list(engine.LABELS), "seed": base_params["seed"],
              "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "compile": True, "bf16": True, "drop_last_per_block": True,
              "historical_data_content_identity_verified": False,
              "inference_runtime_param_overrides": {"distributed": False, "world_size": 1, "rank": 0,
                                                    "local_rank": 0, "gpu": 0},
              "unchanged_data_protocol": {"distributed": True, "world_size": 2, "logical_ranks": [0, 1],
                                           "shuffle": False, "drop_last": True},
              "numeric_equivalence_status": "Requires complete pooled metric check; not assumed from serial scheduling"}
    shared = json.loads(json.dumps(shared))
    engine.prepare_manifest(output, shared, args.resume)
    data_version = engine.canonical_hash(shared["data_files"])
    for variant in args.variants:
        record = records[variant]
        log = ROOT / record["checkpoint_log"]
        checkpoint = log.with_suffix(".model")
        params, historical = engine.parse_log(log)
        engine.validate_parameters(params, base_params, variant)
        if variant != "A":
            params["sisa_variant"] = engine.VARIANTS[variant]
        directory = output / variant
        manifest = {"variant": variant, "variant_name": engine.VARIANTS[variant], "job_task_id": record["job_task_id"],
                    "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": engine.sha256(checkpoint),
                    "checkpoint_file": engine.file_identity(checkpoint), "log_sha256": engine.sha256(log),
                    "historical_metrics": historical, "shared_manifest_sha256": engine.canonical_hash(shared),
                    "data_version": data_version, "logged_params": engine.parse_log(log)[0],
                    "physical_gpus": 1, "logical_world_size": LOGICAL_WORLD_SIZE}
        engine.prepare_manifest(directory, manifest, args.resume)
        if (directory / "complete.json").exists():
            os.replace(directory / "complete.json", directory / "complete.previous.json")
        invariants = serial_variant(params, checkpoint, directory, allocation, data_version,
                                    args.num_workers, args.limit_batches, args.resume)
        if [engine.file_identity(Path(entry["path"])) for entry in shared["data_files"]] != shared["data_files"]:
            raise ValueError("Data files changed during export")
        if engine.file_identity(checkpoint) != manifest["checkpoint_file"]:
            raise ValueError("Checkpoint changed during export")
        rank_receipts = [json.loads((directory / f"rank-{rank}.json").read_text()) for rank in range(LOGICAL_WORLD_SIZE)]
        for rank, entry in enumerate(rank_receipts):
            engine.validate_receipts(entry["shards"], allocation[rank], base_params["batch_size"], args.limit_batches is not None)
        receipts = [receipt for entry in rank_receipts for receipt in entry["shards"]]
        total = engine.validate_receipts(receipts, expected, base_params["batch_size"], args.limit_batches is not None)
        names = {receipt["file"] for receipt in receipts}
        if names != {path.name for path in directory.glob("*.parquet")}:
            raise ValueError("Unaccounted prediction shards")
        identity = {receipt["file"]: receipt["identity_sha256"] for receipt in receipts}
        for completed in output.glob("*/complete.json"):
            if json.loads(completed.read_text())["sample_identity"] != identity:
                raise ValueError(f"Variant sample identity mismatch: {completed}")
        metrics = engine.global_metrics([directory / name for name in sorted(names)], args.limit_batches is not None)
        deviations = {label: {metric: None if metrics[label][metric] is None else
                              metrics[label][metric] - historical[label][metric]
                              for metric in ("AUC", "logloss")} for label in engine.LABELS}
        passed = args.limit_batches is not None or all(
            abs(deviations[label][metric]) <= args.metric_atol for label in engine.LABELS for metric in ("AUC", "logloss"))
        receipt = {"variant": variant, "rows": total, "smoke_only": args.limit_batches is not None,
                   "physical_gpus": 1, "logical_world_size": LOGICAL_WORLD_SIZE,
                   "rows_by_logical_rank": [entry["rows"] for entry in rank_receipts],
                   "metrics": metrics, "historical_deviation": deviations, "metric_check_passed": passed,
                   "sample_identity": identity, "data_version": data_version, "eval_invariants": invariants,
                   "finished_utc": datetime.now(timezone.utc).isoformat()}
        engine.atomic_json(directory / "metrics.json", receipt)
        if not passed:
            raise ValueError(f"Historical metric mismatch; inspect {directory / 'metrics.json'}")
        engine.atomic_json(directory / "complete.json", receipt)
        logging.info("Completed %s serial inference: rows=%d metrics=%s", variant, total, metrics)


if __name__ == "__main__":
    main()
