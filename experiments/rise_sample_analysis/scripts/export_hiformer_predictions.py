"""Strict, sample-identified HiFormer A/E exports on QK-Video or TAAC-25.

The frozen MerRec exporter is imported as a utility engine but never edited.
Only its dataset and label constants are configured in this process; every such
override and both source files are frozen in the output manifest. Worker-side
collation stores labels on the instance, independently of engine globals.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
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
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


ROOT = engine.ROOT
DATASETS = {
    "QK_Video_Action": {"labels": ("click", "follow", "like", "share"), "formal_rows": 48_529_408},
    "TencentGR_10M_Action": {"labels": ("is_click", "is_conversion"), "formal_rows": 153_157_632},
}
ENGINE_DEFAULTS = {"DATASET": "MerRec_Action", "LABELS": ("Like", "Cart", "Offer", "Checkout", "Purchase")}


def configure_engine(dataset_id: str) -> dict:
    if dataset_id not in DATASETS:
        raise ValueError(f"Unsupported HiFormer diagnostic dataset: {dataset_id}")
    engine.DATASET = dataset_id
    engine.LABELS = DATASETS[dataset_id]["labels"]
    return {"DATASET": {"original": ENGINE_DEFAULTS["DATASET"], "effective": engine.DATASET},
            "LABELS": {"original": list(ENGINE_DEFAULTS["LABELS"]), "effective": list(engine.LABELS)}}


def reference_record(dataset_id: str) -> dict:
    source = ROOT / "experiments/sisa_single_seed20262028/results/runs.csv"
    with source.open(newline="") as stream:
        selected = [row for row in csv.DictReader(stream)
                    if row["model"] == "HiFormer" and row["dataset"] == dataset_id]
    if len(selected) != 1 or selected[0]["complete"] != "True":
        raise ValueError(f"Expected one accepted complete HiFormer reference for {dataset_id}")
    return selected[0]


class DatasetCollator:
    """Preserve production input tensors and source IDs across spawned workers."""

    def __init__(self, original, dataset_id: str, labels: tuple[str, ...]):
        self.original = original
        self.dataset_id = dataset_id
        self.labels = tuple(labels)

    def __call__(self, payload):
        batch = self.original(payload)
        batch_dict, item_dict, mask, _ = batch
        size = len(mask)
        meta = {
            "part_id": np.full(size, payload["part_id"], dtype=np.int32),
            "row_offset": np.arange(payload["row_offset"], payload["row_offset"] + size, dtype=np.int64),
            "user_id": batch_dict["user_id"].numpy().reshape(-1).astype(np.int64),
            "item_id": item_dict["item_id"][:, -1].numpy().astype(np.int64),
            "seq_len": batch_dict["seq_len"].numpy().reshape(-1).astype(np.int64),
            "effective_length": mask.sum(dim=1).numpy().astype(np.int16),
        }
        for label in self.labels:
            truth = batch_dict[label].numpy().reshape(-1)
            if not np.isin(truth, [0, 1]).all():
                raise ValueError(f"Nonbinary labels in {self.dataset_id}: {label}")
            meta[label] = truth.astype(np.uint8)
        if np.any(meta["effective_length"] > np.minimum(meta["seq_len"], self.original.max_len)):
            raise ValueError("Invalid effective history length")
        return batch, meta


def build_loader(params: dict, rank: int, world_size: int, num_workers: int):
    dataset_id = params["dataset_id"]
    labels = DATASETS[dataset_id]["labels"]
    feature_map = engine.feature_map_from_config(params)
    if tuple(feature_map.labels) != labels:
        raise ValueError("Historical feature-map label order differs from dataset protocol")
    dataset = engine.IdentifiedDataset(
        params["test_data"], params["test_user_info"], params["test_item_info"],
        batch_size=params["batch_size"], shuffle=False, distributed=world_size > 1,
        rank=rank, world_size=world_size, drop_last=True, seed=params["dataloader_seed"],
    )
    original = engine.BlockedBatchCollator(
        feature_map, params["max_len"], dataset.column_index,
        params["test_user_info"], params["test_item_info"], cache_size=params["block_cache_size"],
    )
    kwargs = dict(dataset=dataset, batch_size=None, shuffle=False, num_workers=num_workers,
                  collate_fn=DatasetCollator(original, dataset_id, labels), pin_memory=True,
                  persistent_workers=num_workers > 0)
    if num_workers:
        kwargs.update(prefetch_factor=4, multiprocessing_context="spawn")
    return dataset, DataLoader(**kwargs)


def coverage_by_rank(dataset, world_size: int) -> list[dict[int, int]]:
    buckets, _ = dataset._assign_blocks_greedily(dataset.blocks, world_size)
    return [{int(block["part_id"]): int(block["num_rows"]) // dataset.batch_size * dataset.batch_size
             for block in bucket} for bucket in buckets]


def validate_coverage(dataset_id: str, rank_coverage: list[dict[int, int]], formal: bool) -> dict[int, int]:
    expected = {}
    for per_rank in rank_coverage:
        if set(per_rank) & set(expected):
            raise ValueError("A source part was assigned to multiple ranks")
        expected.update(per_rank)
    if formal and (len(rank_coverage) != 2 or sum(expected.values()) != DATASETS[dataset_id]["formal_rows"]):
        raise ValueError("Historical global test coverage no longer matches")
    return expected


def load_model(params: dict, checkpoint: Path, device: int):
    from model_zoo.HiFormer import HiFormer
    configured = dict(params, gpu=device)
    feature_map = engine.feature_map_from_config(configured)
    model = HiFormer(feature_map, **configured)
    if not model._torch_compile_enabled:
        raise RuntimeError("Historical checkpoint requires successful compile wrapping")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    model.optimizers = []
    model.optimizer = model.dense_optimizer = model.sparse_optimizer = None
    model.eval()
    return model


def source_hashes() -> dict:
    hashes = engine.source_hashes()
    hashes[str(Path(__file__).relative_to(ROOT))] = engine.sha256(Path(__file__))
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--variants", nargs="+", choices=["A", "E"], default=["E"])
    parser.add_argument("--extra-baseline-log", type=Path)
    parser.add_argument("--baseline-task-id", default="not_recorded")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-batches", type=int, help="Smoke only: batches per rank and variant")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--metric-atol", type=float, default=1e-6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit_batches is not None and args.limit_batches < 1:
        parser.error("--limit-batches must be positive")
    if args.num_workers < 0 or args.metric_atol < 0 or len(set(args.variants)) != len(args.variants):
        parser.error("Invalid workers, tolerance, or duplicate variants")
    if ("A" in args.variants) != (args.extra_baseline_log is not None):
        parser.error("A and --extra-baseline-log must be supplied together")
    overrides = configure_engine(args.dataset)
    labels = DATASETS[args.dataset]["labels"]
    os.chdir(ROOT)
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "artifacts/rise_sample_analysis"):
        parser.error("Output must be below artifacts/rise_sample_analysis")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2 and args.limit_batches is None:
        parser.error("Formal inference requires torchrun --nproc_per_node=2")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; CPU tests never load historical weights")
    torch.cuda.set_device(local_rank)
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO, format=f"rank={rank} %(asctime)s %(message)s")
    if world_size > 1:
        dist.init_process_group("gloo", timeout=timedelta(hours=8))
    output_lock = None
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        output_lock = (output / ".export.lock").open("a")
        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records = {"E": reference_record(args.dataset)}
    if args.extra_baseline_log is not None:
        log = engine.validate_extra_log(args.extra_baseline_log)
        records["A"] = {"checkpoint_log": str(log.relative_to(ROOT)), "job_task_id": args.baseline_task_id}
    base_params, _ = engine.parse_log(ROOT / records["E"]["checkpoint_log"])
    required = {"model": "HiFormer", "dataset_id": args.dataset, "seed": 20262028,
                "world_size": 2, "batch_size": 16384, "max_len": 100,
                "enable_bf16": True, "enable_torch_compile": True}
    if any(base_params.get(key) != value for key, value in required.items()):
        raise ValueError("Unexpected HiFormer reference protocol")
    dataset, loader = build_loader(base_params, rank, world_size, args.num_workers)
    allocation = coverage_by_rank(dataset, world_size)
    expected = validate_coverage(args.dataset, allocation, args.limit_batches is None)
    if dataset.num_batches * base_params["batch_size"] != sum(allocation[rank].values()):
        raise ValueError("Loader rank allocation disagrees with declared coverage")
    shared = None
    if rank == 0:
        files = sorted({str(block[key]) for block in dataset.blocks
                        for key in ("data_file", "user_info_file", "item_info_file")}
                       | {str(engine.find_meta_data_json(base_params["test_user_info"]))})
        shared = {"schema_version": engine.SCHEMA_VERSION, "dataset": args.dataset, "split": "test",
                  "model": "HiFormer", "world_size": world_size, "batch_size": base_params["batch_size"],
                  "limit_batches_per_rank": args.limit_batches, "num_workers": args.num_workers,
                  "metric_atol": args.metric_atol, "data_files": [engine.file_identity(Path(path)) for path in files],
                  "source_sha256": source_hashes(), "engine_constant_overrides": overrides,
                  "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                  "expected_rows_by_part": expected, "expected_rows_by_rank": [sum(parts.values()) for parts in allocation],
                  "expected_parts_by_rank": allocation, "labels": list(labels), "seed": base_params["seed"],
                  "float32_matmul_precision": torch.get_float32_matmul_precision(),
                  "compile": True, "bf16": True, "drop_last_per_block": True,
                  "historical_data_content_identity_verified": False}
        shared = json.loads(json.dumps(shared))
        engine.prepare_manifest(output, shared, args.resume)
    if world_size > 1:
        objects = [shared]
        dist.broadcast_object_list(objects, src=0)
        shared = objects[0]
    data_version = engine.canonical_hash(shared["data_files"])
    for variant in args.variants:
        row = records[variant]
        log = ROOT / row["checkpoint_log"]
        checkpoint = log.with_suffix(".model")
        params, historical = engine.parse_log(log)
        engine.validate_parameters(params, base_params, variant)
        if variant == "E":
            params["sisa_variant"] = "full"
        params.update(rank=rank, local_rank=local_rank, world_size=world_size, distributed=world_size > 1)
        directory = output / variant
        if rank == 0:
            manifest = {"variant": variant, "variant_name": engine.VARIANTS[variant], "job_task_id": row["job_task_id"],
                        "model": "HiFormer", "dataset": args.dataset,
                        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": engine.sha256(checkpoint),
                        "checkpoint_file": engine.file_identity(checkpoint), "log_sha256": engine.sha256(log),
                        "historical_metrics": historical, "shared_manifest_sha256": engine.canonical_hash(shared),
                        "data_version": data_version, "logged_params": engine.parse_log(log)[0]}
            engine.prepare_manifest(directory, manifest, args.resume)
            if (directory / "complete.json").exists():
                os.replace(directory / "complete.json", directory / "complete.previous.json")
        engine.barrier()
        engine.seed_everything(params["seed"] + rank)
        model = None
        receipts = []
        try:
            for index, (batch, meta) in enumerate(loader):
                if args.limit_batches is not None and index >= args.limit_batches:
                    break
                path = directory / engine.shard_name(meta)
                if args.resume and path.exists():
                    if path.with_suffix(".json").exists():
                        receipts.append(engine.verify_shard(path, engine.identity_hash(meta)))
                        continue
                    os.replace(path, path.with_name(path.name + f".uncommitted-{os.getpid()}"))
                if model is None:
                    model = load_model(params, checkpoint, local_rank)
                with torch.no_grad(), model._get_amp_context():
                    predictions = model.forward(batch)
                values = {label: predictions[f"{label}_pred"].float().detach().cpu().numpy().reshape(-1)
                          for label in labels}
                receipts.append(engine.write_shard(directory, meta, values))
                if index % 20 == 0:
                    logging.info("dataset=%s variant=%s batch=%d", args.dataset, variant, index)
            local_rows = engine.validate_receipts(receipts, allocation[rank], base_params["batch_size"],
                                                  args.limit_batches is not None)
            engine.atomic_json(directory / f"rank-{rank}.json", {
                "rank": rank, "shards": receipts, "rows": local_rows, "data_version": data_version})
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        engine.barrier()
        if rank == 0:
            if [engine.file_identity(Path(entry["path"])) for entry in shared["data_files"]] != shared["data_files"]:
                raise ValueError("Data files changed during inference")
            if engine.file_identity(checkpoint) != manifest["checkpoint_file"]:
                raise ValueError("Checkpoint changed during inference")
            rank_receipts = [json.loads((directory / f"rank-{r}.json").read_text()) for r in range(world_size)]
            for r, entry in enumerate(rank_receipts):
                engine.validate_receipts(entry["shards"], allocation[r], base_params["batch_size"],
                                         args.limit_batches is not None)
            all_receipts = [receipt for entry in rank_receipts for receipt in entry["shards"]]
            total = engine.validate_receipts(all_receipts, expected, base_params["batch_size"], args.limit_batches is not None)
            names = {receipt["file"] for receipt in all_receipts}
            if names != {path.name for path in directory.glob("*.parquet")}:
                raise ValueError("Unaccounted output shards")
            identity = {receipt["file"]: receipt["identity_sha256"] for receipt in all_receipts}
            for completed in output.glob("*/complete.json"):
                if json.loads(completed.read_text())["sample_identity"] != identity:
                    raise ValueError(f"Different paired sample identity: {completed}")
            metrics = engine.global_metrics([directory / name for name in sorted(names)], args.limit_batches is not None)
            deviations = {label: {metric: None if metrics[label][metric] is None else
                                  metrics[label][metric] - historical[label][metric]
                                  for metric in ("AUC", "logloss")} for label in labels}
            passed = args.limit_batches is not None or all(
                abs(deviations[label][metric]) <= args.metric_atol for label in labels for metric in ("AUC", "logloss"))
            receipt = {"model": "HiFormer", "dataset": args.dataset, "variant": variant, "rows": total,
                       "rows_by_rank": [entry["rows"] for entry in rank_receipts],
                       "smoke_only": args.limit_batches is not None, "metrics": metrics,
                       "historical_deviation": deviations, "metric_check_passed": passed,
                       "sample_identity": identity, "data_version": data_version,
                       "finished_utc": datetime.now(timezone.utc).isoformat()}
            engine.atomic_json(directory / "metrics.json", receipt)
            if not passed:
                raise ValueError(f"Historical metric mismatch; inspect {directory / 'metrics.json'}")
            engine.atomic_json(directory / "complete.json", receipt)
            logging.info("Completed %s %s: rows=%d metrics=%s", args.dataset, variant, total, metrics)
        engine.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
