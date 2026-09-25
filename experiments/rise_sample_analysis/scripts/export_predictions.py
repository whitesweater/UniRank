"""Export existing HyFormer--MerRec component checkpoints without training.

Formal example (run from the repository environment):
torchrun --standalone --nproc_per_node=2 experiments/rise_sample_analysis/scripts/export_predictions.py \
    --variants B C D E --output-dir artifacts/rise_sample_analysis/predictions/formal

A smoke run requires --limit-batches and a separate output directory. Every
output is immutable unless --resume verifies its manifest and shard receipts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, get_worker_info

from unirank.features import feature_map_from_config
from unirank.metrics import evaluate_metrics
from unirank.pytorch.dataloaders.unirank_dataloader import (
    BlockedBatchCollator,
    BlockedParquetBatchDataset,
)
from unirank.pytorch.torch_utils import seed_everything
from unirank.utils import dataframe_to_darray, find_meta_data_json

DATASET = "MerRec_Action"
LABELS = ("Like", "Cart", "Offer", "Checkout", "Purchase")
VARIANTS = {"A": "native_baseline", "B": "content_only", "C": "decay_only", "D": "rotation_only", "E": "full"}
COMPONENTS = ("B", "C", "D", "E")
FORMAL_ROWS = 11_599_872
FORMAL_RANK_ROWS = 5_799_936
SCHEMA_VERSION = 1
META_COLUMNS = ("part_id", "row_offset", "user_id", "item_id", "seq_len", "effective_length")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_log(path: Path) -> tuple[dict, dict]:
    text = path.read_text()
    raw, _ = json.JSONDecoder().raw_decode(text.split("Params: ", 1)[1])
    params = {}
    for key, value in raw.items():
        try:
            params[key] = ast.literal_eval(value) if isinstance(value, str) else value
        except (ValueError, SyntaxError):
            params[key] = value
    if text.count("Test evaluation") != 1:
        raise ValueError(f"Expected exactly one final evaluation: {path}")
    tail = text.rsplit("Test evaluation", 1)[1]
    metrics = {
        label: {name: float(value) for name, value in re.findall(r"(AUC|logloss): ([\d.]+)", body)}
        for label, body in re.findall(r"\[Task: (\w+)\]\[Metrics\] (.*)", tail)
    }
    if set(metrics) != set(LABELS):
        raise ValueError(f"Incomplete historical labels: {path}")
    return params, metrics


def run_records() -> dict[str, dict]:
    path = ROOT / "experiments/ablations/sisa_component_report/archive/results/runs.csv"
    with path.open(newline="") as stream:
        rows = {row["setting"]: row for row in csv.DictReader(stream)
                if row["model"] == "HyFormer" and row["dataset"] == DATASET}
    if set(rows) != set(COMPONENTS):
        raise ValueError("Expected four fixed component records")
    return rows


def validate_parameters(params: dict, reference: dict, variant: str) -> None:
    differences = {key for key in params.keys() | reference.keys()
                   if params.get(key) != reference.get(key)}
    allowed = {"model_id", "sisa_variant"}
    if variant == "A":
        if params.get("sisa_enabled", False):
            raise ValueError("A must be a independently trained native baseline with SISA disabled")
        # model_root is only an output location, not an inference/training control.
        allowed.add("model_root")
        allowed |= {key for key in differences if key.startswith("sisa_")}
    elif not params.get("sisa_enabled", False):
        raise ValueError(f"Component {variant} requires enabled SISA")
    if differences - allowed:
        raise ValueError(f"Historical parameter mismatch: {sorted(differences - allowed)}")


def validate_extra_log(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (ROOT / "checkpoints" / DATASET,
               ROOT / "artifacts/rise_sample_analysis/training_checkpoints" / DATASET)
    if not any(resolved.is_relative_to(directory.resolve()) for directory in allowed):
        raise ValueError("Extra baseline log must be under a declared MerRec checkpoint directory")
    return resolved


class IdentifiedDataset(BlockedParquetBatchDataset):
    """Original deterministic test batching, with row offsets attached at source.

    Rank assignment, cost estimates, column encoding and worker block partition
    come from the production loader. No shuffled-input mode is supported.
    """

    def __iter__(self):
        if self.shuffle or not self.drop_last:
            raise ValueError("Export requires shuffle=False and per-block drop_last=True")
        blocks = list(self.rank_blocks)
        worker = get_worker_info()
        if worker is not None:
            blocks = blocks[worker.id::worker.num_workers]
        for block in blocks:
            frame = pd.read_parquet(block["data_file"], columns=self.columns)
            array, column_index = dataframe_to_darray(frame)
            if column_index != self.column_index:
                raise ValueError(f"Column layout drift: {block['data_file']}")
            for start in range(0, len(array) - self.batch_size + 1, self.batch_size):
                yield {
                    "rows": np.ascontiguousarray(array[start:start + self.batch_size]),
                    "part_id": int(block["part_id"]),
                    "row_offset": start,
                    "data_file": block["data_file"],
                    "user_info_file": block["user_info_file"],
                    "item_info_file": block["item_info_file"],
                }


class IdentifiedCollator:
    def __init__(self, original: BlockedBatchCollator):
        self.original = original

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
        for label in LABELS:
            meta[label] = batch_dict[label].numpy().reshape(-1).astype(np.uint8)
            if not np.isin(batch_dict[label].numpy(), [0, 1]).all():
                raise ValueError(f"Nonbinary labels: {label}")
        if np.any(meta["effective_length"] > np.minimum(meta["seq_len"], self.original.max_len)):
            raise ValueError("Invalid effective history length")
        return batch, meta


def identity_hash(meta: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256((DATASET + "/test").encode())
    for column in META_COLUMNS + LABELS:
        digest.update(column.encode())
        digest.update(np.asarray(meta[column], dtype="<i8").tobytes())
    return digest.hexdigest()


def shard_name(meta: dict[str, np.ndarray]) -> str:
    part = np.asarray(meta["part_id"])
    offsets = np.asarray(meta["row_offset"])
    if len(offsets) == 0 or not np.all(part == part[0]) or not np.array_equal(
            offsets, np.arange(offsets[0], offsets[0] + len(offsets))):
        raise ValueError("Shard must be one contiguous, nonempty original block interval")
    return f"part-{int(part[0]):05d}-offset-{int(offsets[0]):012d}.parquet"


def verify_shard(path: Path, expected_identity: str | None = None) -> dict:
    receipt = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != receipt["sha256"]:
        raise ValueError(f"Corrupt or changed shard: {path}")
    if expected_identity is not None and receipt["identity_sha256"] != expected_identity:
        raise ValueError(f"Sample/label identity changed: {path}")
    if pq.ParquetFile(path).metadata.num_rows != receipt["rows"]:
        raise ValueError(f"Invalid shard row count: {path}")
    return receipt


def write_shard(directory: Path, meta: dict, predictions: dict) -> dict:
    path = directory / shard_name(meta)
    receipt_path = path.with_suffix(".json")
    if path.exists() or receipt_path.exists():
        raise FileExistsError(f"Refusing to overwrite shard: {path}")
    size = len(meta["row_offset"])
    columns = {"dataset": pa.array([DATASET] * size), "split": pa.array(["test"] * size)}
    columns.update({name: pa.array(value) for name, value in meta.items()})
    for label in LABELS:
        prediction = np.asarray(predictions[label], dtype=np.float32).reshape(-1)
        if len(prediction) != size or not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 1)):
            raise ValueError(f"Invalid probabilities: {label}")
        columns[f"{label}_pred"] = pa.array(prediction)
    table = pa.table(columns)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    pq.write_table(table, temporary, compression="zstd")
    receipt = {"file": path.name, "rows": size, "part_id": int(meta["part_id"][0]),
               "row_start": int(meta["row_offset"][0]), "identity_sha256": identity_hash(meta),
               "sha256": sha256(temporary)}
    os.replace(temporary, path)
    atomic_json(receipt_path, receipt)
    return receipt


def prepare_manifest(directory: Path, manifest: dict, resume: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        if not resume:
            raise FileExistsError(f"Output exists; use a new directory or --resume: {directory}")
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f"Resume manifest differs: {path}")
    else:
        if any(path.name != ".export.lock" for path in directory.iterdir()):
            raise ValueError(f"Nonempty directory has no manifest: {directory}")
        atomic_json(path, manifest)


def global_metrics(paths: list[Path], allow_single_class: bool = False) -> dict:
    """Compute metrics after pooling every rank, never average rank AUCs."""
    result = {}
    for label in LABELS:
        truth, prediction = [], []
        for path in paths:
            table = pq.read_table(path, columns=[label, f"{label}_pred"])
            truth.append(table[label].to_numpy().astype(np.float64))
            prediction.append(table[f"{label}_pred"].to_numpy().astype(np.float64))
        y_true, y_pred = np.concatenate(truth), np.concatenate(prediction)
        if len(np.unique(y_true)) < 2:
            if not allow_single_class:
                raise ValueError(f"Single-class formal evaluation: {label}")
            result[label] = {"AUC": None, "logloss": None, "positive": int(y_true.sum())}
        else:
            result[label] = dict(evaluate_metrics(y_true, y_pred, ["AUC", "logloss"]))
            result[label]["positive"] = int(y_true.sum())
        result[label]["rows"] = len(y_true)
    return result


def validate_receipts(receipts: list[dict], expected: dict[int, int], batch_size: int,
                      limited: bool) -> int:
    by_part: dict[int, list[tuple[int, int]]] = {}
    for receipt in receipts:
        by_part.setdefault(receipt["part_id"], []).append((receipt["row_start"], receipt["rows"]))
    if not set(by_part).issubset(expected):
        raise ValueError("Unexpected exported part")
    total = 0
    for part, intervals in by_part.items():
        endpoint = 0
        for start, size in sorted(intervals):
            if start != endpoint or size != batch_size:
                raise ValueError(f"Duplicate, gap, or partial batch in part {part}")
            endpoint += size
        if endpoint > expected[part] or (not limited and endpoint != expected[part]):
            raise ValueError(f"Coverage mismatch in part {part}")
        total += endpoint
    if not limited and set(by_part) != {part for part, rows in expected.items() if rows}:
        raise ValueError("Missing part")
    return total


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_hashes() -> dict:
    paths = [Path(__file__).resolve(), ROOT / "uv.lock", ROOT / "pyproject.toml"]
    paths += sorted((ROOT / "unirank").rglob("*.py"))
    paths += sorted((ROOT / "model_zoo").rglob("*.py"))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def load_model(params: dict, checkpoint: Path, device: int):
    from model_zoo.HyFormer import HyFormer
    configured = dict(params, gpu=device)
    feature_map = feature_map_from_config(configured)
    model = HyFormer(feature_map, **configured)
    if not model._torch_compile_enabled:
        raise RuntimeError("Historical compiled checkpoint requires successful compile wrapping")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    # Inference never steps an optimizer. Discard its eager Adagrad state.
    model.optimizers = []
    model.optimizer = model.dense_optimizer = model.sparse_optimizer = None
    model.eval()
    return model


def build_loader(params: dict, rank: int, world_size: int, num_workers: int):
    feature_map = feature_map_from_config(params)
    dataset = IdentifiedDataset(
        params["test_data"], params["test_user_info"], params["test_item_info"],
        batch_size=params["batch_size"], shuffle=False, distributed=world_size > 1,
        rank=rank, world_size=world_size, drop_last=True, seed=params["dataloader_seed"],
    )
    collator = IdentifiedCollator(BlockedBatchCollator(
        feature_map, params["max_len"], dataset.column_index,
        params["test_user_info"], params["test_item_info"], cache_size=params["block_cache_size"],
    ))
    kwargs = dict(dataset=dataset, batch_size=None, shuffle=False, num_workers=num_workers,
                  collate_fn=collator, pin_memory=True, persistent_workers=num_workers > 0)
    if num_workers:
        kwargs.update(prefetch_factor=4, multiprocessing_context="spawn")
    return dataset, DataLoader(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(COMPONENTS))
    parser.add_argument("--extra-baseline-log", type=Path,
                        help="Completed matched HyFormer--MerRec native baseline log for --variants A")
    parser.add_argument("--baseline-task-id", default="not_recorded",
                        help="Scheduler job/task identity of the extra baseline, if known")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-batches", type=int, help="Smoke only: batches per rank and variant")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--metric-atol", type=float, default=1e-6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit_batches is not None and args.limit_batches < 1:
        parser.error("--limit-batches must be positive")
    if args.num_workers < 0 or args.metric_atol < 0:
        parser.error("workers and tolerance must be nonnegative")
    if len(set(args.variants)) != len(args.variants):
        parser.error("duplicate variants")
    if ("A" in args.variants) != (args.extra_baseline_log is not None):
        parser.error("--variants A requires --extra-baseline-log (and vice versa)")
    os.chdir(ROOT)
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "artifacts/rise_sample_analysis"):
        parser.error("output must be below artifacts/rise_sample_analysis")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2 and args.limit_batches is None:
        parser.error("Formal export requires torchrun --nproc_per_node=2")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; CPU tests do not load real checkpoints")
    torch.cuda.set_device(local_rank)
    # run_expid.py sets this before model construction; retain its TF32 policy.
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO, format=f"rank={rank} %(asctime)s %(message)s")
    if world_size > 1:
        dist.init_process_group("gloo", timeout=timedelta(hours=4))
    output_lock = None
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        output_lock = (output / ".export.lock").open("a")
        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records = run_records()
    if args.extra_baseline_log is not None:
        extra_log = validate_extra_log(args.extra_baseline_log)
        records["A"] = {"checkpoint_log": str(extra_log.relative_to(ROOT)),
                        "job_task_id": args.baseline_task_id}
    base_params, _ = parse_log(ROOT / records["E"]["checkpoint_log"])
    dataset, loader = build_loader(base_params, rank, world_size, args.num_workers)
    expected = {int(block["part_id"]): int(block["num_rows"]) // base_params["batch_size"] * base_params["batch_size"]
                for block in dataset.blocks}
    if args.limit_batches is None and (
            sum(expected.values()) != FORMAL_ROWS or dataset.num_batches * base_params["batch_size"] != FORMAL_RANK_ROWS):
        raise ValueError("Historical sample coverage no longer matches")
    shared = None
    if rank == 0:
        files = sorted({str(block[key]) for block in dataset.blocks
                        for key in ("data_file", "user_info_file", "item_info_file")}
                       | {str(find_meta_data_json(base_params["test_user_info"]))})
        shared = {"schema_version": SCHEMA_VERSION, "dataset": DATASET, "split": "test",
                  "world_size": world_size, "batch_size": base_params["batch_size"],
                  "limit_batches_per_rank": args.limit_batches, "num_workers": args.num_workers,
                  "metric_atol": args.metric_atol, "data_files": [file_identity(Path(path)) for path in files],
                  "source_sha256": source_hashes(), "torch_version": torch.__version__,
                  "cuda_version": torch.version.cuda, "expected_rows_by_part": expected,
                  "labels": list(LABELS), "seed": base_params["seed"],
                  "float32_matmul_precision": torch.get_float32_matmul_precision(),
                  "compile": True, "bf16": True, "drop_last_per_block": True,
                  "historical_data_content_identity_verified": False}
        shared = json.loads(json.dumps(shared))
        prepare_manifest(output, shared, args.resume)
    if world_size > 1:
        objects = [shared]
        dist.broadcast_object_list(objects, src=0)
        shared = objects[0]
    data_version = canonical_hash(shared["data_files"])
    for variant in args.variants:
        row = records[variant]
        log = ROOT / row["checkpoint_log"]
        checkpoint = log.with_suffix(".model")
        params, historical = parse_log(log)
        validate_parameters(params, base_params, variant)
        if variant != "A":
            params["sisa_variant"] = VARIANTS[variant]
        params.update(rank=rank, local_rank=local_rank, world_size=world_size, distributed=world_size > 1)
        directory = output / variant
        if rank == 0:
            manifest = {"variant": variant, "variant_name": VARIANTS[variant], "job_task_id": row["job_task_id"],
                        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
                        "checkpoint_file": file_identity(checkpoint),
                        "log_sha256": sha256(log), "historical_metrics": historical,
                        "shared_manifest_sha256": canonical_hash(shared), "data_version": data_version,
                        "logged_params": parse_log(log)[0]}
            prepare_manifest(directory, manifest, args.resume)
            if (directory / "complete.json").exists():
                os.replace(directory / "complete.json", directory / "complete.previous.json")
        barrier()
        seed_everything(params["seed"] + rank)
        model = None
        receipts = []
        try:
            for index, (batch, meta) in enumerate(loader):
                if args.limit_batches is not None and index >= args.limit_batches:
                    break
                path = directory / shard_name(meta)
                if args.resume and path.exists():
                    if not path.with_suffix(".json").exists():
                        # Interrupted before atomic receipt: retain bytes for audit,
                        # then recompute this shard instead of accepting it silently.
                        orphan = path.with_name(path.name + f".uncommitted-{os.getpid()}")
                        os.replace(path, orphan)
                    else:
                        receipts.append(verify_shard(path, identity_hash(meta)))
                        continue
                if model is None:
                    model = load_model(params, checkpoint, local_rank)
                with torch.no_grad(), model._get_amp_context():
                    predictions = model.forward(batch)
                values = {label: predictions[f"{label}_pred"].float().detach().cpu().numpy().reshape(-1)
                          for label in LABELS}
                receipts.append(write_shard(directory, meta, values))
                if index % 20 == 0:
                    logging.info("variant=%s batch=%d rows_written=%d", variant, index, sum(r["rows"] for r in receipts))
            atomic_json(directory / f"rank-{rank}.json", {"rank": rank, "shards": receipts,
                        "rows": sum(receipt["rows"] for receipt in receipts), "data_version": data_version})
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        barrier()
        if rank == 0:
            if [file_identity(Path(entry["path"])) for entry in shared["data_files"]] != shared["data_files"]:
                raise ValueError("Data file metadata changed during export")
            if file_identity(checkpoint) != manifest["checkpoint_file"]:
                raise ValueError("Checkpoint file changed during export")
            rank_receipts = [json.loads((directory / f"rank-{r}.json").read_text()) for r in range(world_size)]
            all_receipts = [receipt for entry in rank_receipts for receipt in entry["shards"]]
            total = validate_receipts(all_receipts, expected, base_params["batch_size"], args.limit_batches is not None)
            names = {receipt["file"] for receipt in all_receipts}
            if names != {path.name for path in directory.glob("*.parquet")}:
                raise ValueError("Unaccounted output shards")
            # Existing variants must describe the same IDs, labels and metadata.
            identity = {receipt["file"]: receipt["identity_sha256"] for receipt in all_receipts}
            for completed in output.glob("*/complete.json"):
                other = json.loads(completed.read_text())
                if other["sample_identity"] != identity:
                    raise ValueError(f"Different paired sample identity: {completed}")
            metrics = global_metrics([directory / name for name in sorted(names)], args.limit_batches is not None)
            deviations = {label: {metric: None if metrics[label][metric] is None else
                                  metrics[label][metric] - historical[label][metric]
                                  for metric in ("AUC", "logloss")} for label in LABELS}
            passed = args.limit_batches is not None or all(
                abs(deviations[label][metric]) <= args.metric_atol
                for label in LABELS for metric in ("AUC", "logloss"))
            receipt = {"variant": variant, "rows": total, "smoke_only": args.limit_batches is not None,
                       "metrics": metrics, "historical_deviation": deviations,
                       "metric_check_passed": passed, "sample_identity": identity,
                       "data_version": data_version, "finished_utc": datetime.now(timezone.utc).isoformat()}
            atomic_json(directory / "metrics.json", receipt)
            if not passed:
                raise ValueError(f"Historical metric mismatch; inspect {directory / 'metrics.json'}")
            atomic_json(directory / "complete.json", receipt)
            logging.info("Completed variant=%s rows=%d metrics=%s", variant, total, metrics)
        barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
