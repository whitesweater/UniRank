"""Frozen-checkpoint HiFormer/TencentGR target-attention diagnostic (no training)."""

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

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "experiments/rise_sample_analysis/scripts"
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

import export_hiformer_predictions as wrapper
from observe_mechanism import device_metadata, tree_map
from unirank.pytorch.layers.attentions.dot_product_attention import ScaledDotProductAttention

DATASET = "TencentGR_10M_Action"
LABELS = ("is_click", "is_conversion")
SITE = "attention_layers.dot_attention"
engine = wrapper.engine


def selected_offsets(part: int, window: int, count: int, seed: int) -> np.ndarray:
    if not 0 < count <= window:
        raise ValueError("Invalid sampling size")
    priorities = [hashlib.sha256(f"{seed}:{part}:{row}".encode()).digest() for row in range(window)]
    return np.array(sorted(sorted(range(window), key=priorities.__getitem__)[:count]), dtype=np.int64)


def prediction_comparison(actual, reference, *, atol=0.0, rtol=0.0):
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("Prediction tolerances must be finite and nonnegative")
    result = {}
    for label in LABELS:
        a, b = np.asarray(actual[label]), np.asarray(reference[label])
        if a.size == 0 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Invalid prediction shape or nonfinite value")
        result[label] = {"bitwise_equal": bool(np.array_equal(a, b)),
                         "max_abs_error": float(np.max(np.abs(a - b))),
                         "passed": bool(np.allclose(a, b, atol=atol, rtol=rtol))}
    return {"atol": atol, "rtol": rtol, "labels": result,
            "bitwise_equal": all(x["bitwise_equal"] for x in result.values()),
            "passed": all(x["passed"] for x in result.values())}


def reconstruct_attention(query, key, delta, mask, scale):
    """FP32 reconstruction on the complete native allowed set; empty rows remain NaN."""
    if query.ndim != 4 or query.shape[-2] != 1 or mask is None or delta is None:
        raise ValueError("Expected masked single-query auxiliary attention")
    if not bool(torch.all((mask == 0) | (mask == 1))):
        raise ValueError("Native mask must be binary")
    divisor = float(scale) if scale is not None else query.shape[-1] ** 0.5
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("Attention scale must be finite and positive")
    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = query.float() @ key.float().transpose(-2, -1)
        scores = scores / divisor
        correction = delta.to(query.dtype).float().expand_as(scores)
        allowed = mask.bool().expand_as(scores)
        nonempty = allowed.any(-1)
        if not torch.isfinite(scores).all() or not torch.isfinite(correction).all():
            raise ValueError("Nonfinite captured scores")
        before_logits = scores.masked_fill(~allowed, -torch.inf)
        after_logits = (scores + correction).masked_fill(~allowed, -torch.inf)
        before_logits = torch.where(nonempty[..., None], before_logits, torch.zeros_like(scores))
        after_logits = torch.where(nonempty[..., None], after_logits, torch.zeros_like(scores))
        before = before_logits.softmax(-1).masked_fill(~nonempty[..., None], torch.nan)
        after = after_logits.softmax(-1).masked_fill(~nonempty[..., None], torch.nan)
        for weights in (before, after):
            if not torch.allclose(weights.sum(-1)[nonempty], torch.ones_like(weights.sum(-1)[nonempty]), atol=1e-6, rtol=0):
                raise ValueError("Attention normalization failed")
            if torch.any(weights[~allowed & nonempty[..., None]] != 0):
                raise ValueError("Padding probability is nonzero")
        tv = (before - after).abs().sum(-1) / 2
        omega = correction.masked_fill(~allowed, -torch.inf).amax(-1) - correction.masked_fill(~allowed, torch.inf).amin(-1)
        omega = omega.masked_fill(~nonempty, torch.nan)
        if torch.any(tv[nonempty] > torch.tanh(omega[nonempty] / 4) + 1e-6):
            raise ValueError("TV bound failed")
    result = {"native_scores": scores.squeeze(-2), "delta": correction.squeeze(-2),
              "weight_before": before.squeeze(-2), "weight_after": after.squeeze(-2),
              "tv": tv.squeeze(-1), "omega": omega.squeeze(-1)}
    return {name: value.detach().float().cpu().numpy() for name, value in result.items()}


class Capture:
    def __init__(self, model):
        self.entries = []
        matches = [(name, module) for name, module in model.named_modules()
                   if isinstance(module, ScaledDotProductAttention)
                   and ".".join(x for x in name.split(".") if x != "_orig_mod") == SITE]
        if len(matches) != 1 or matches[0][1].attention_activation_type != "SoftMax":
            raise ValueError(f"Expected one target softmax site, found {len(matches)}")
        self.handle = matches[0][1].register_forward_pre_hook(self.hook, with_kwargs=True)

    def hook(self, module, args, kwargs):
        self.entries.append((args, kwargs))

    def consume(self, selected_indices):
        if len(self.entries) != 1:
            raise ValueError(f"Expected exactly one capture, found {len(self.entries)}")
        args, kwargs = self.entries.pop()
        indices = torch.as_tensor(selected_indices, dtype=torch.long, device=args[0].device)
        def selected(tensor):
            if tensor is None:
                return None
            return tensor.index_select(0, indices.to(tensor.device))
        return reconstruct_attention(selected(args[0]), selected(args[1]), selected(kwargs.get("score_bias")),
                                     selected(kwargs.get("mask")), kwargs.get("scale"))

    def close(self):
        self.handle.remove()
        self.entries.clear()


def save_sample_cache(path, samples):
    torch.save([{"batch": sample["batch"],
                 "selected_indices": torch.from_numpy(sample["selected_indices"]),
                 "meta": {key: torch.from_numpy(value) for key, value in sample["meta"].items()}}
                for sample in samples], path)


def collect_samples(params, count, seed, output):
    dataset, loader = wrapper.build_loader(params, 0, 2, 0)
    del loader
    coverage = wrapper.validate_coverage(DATASET, wrapper.coverage_by_rank(dataset, 2), True)
    blocks = sorted(dataset.blocks, key=lambda b: int(b["part_id"]))
    if len(blocks) != 8:
        raise ValueError("Expected exactly eight physical test parts")
    collator = engine.BlockedBatchCollator(engine.feature_map_from_config(params), params["max_len"],
                                         dataset.column_index, params["test_user_info"],
                                         params["test_item_info"], cache_size=1)
    samples, frame, files = [], [], {}
    for block in blocks:
        part = int(block["part_id"])
        window = params["batch_size"]
        if coverage[part] < window:
            raise ValueError("Sampling window exceeds retained formal coverage")
        for field in ("data_file", "user_info_file", "item_info_file"):
            files[str(block[field])] = engine.file_identity(Path(block[field]))
        logging.info("Sampling part %s", part)
        table = next(pq.ParquetFile(block["data_file"]).iter_batches(batch_size=window, columns=dataset.columns))
        if table.num_rows != window:
            raise ValueError("Incomplete fixed sampling window")
        rows, layout = engine.dataframe_to_darray(table.to_pandas())
        if layout != dataset.column_index:
            raise ValueError("Input column layout changed")
        offsets = selected_offsets(part, window, count, seed)
        payload = dict(block, rows=np.ascontiguousarray(rows))
        batch = collator(payload)
        batch_dict, item_dict, mask, _ = batch
        meta = {"part_id": np.full(count, part, dtype=np.int32), "row_offset": offsets,
                "user_id": batch_dict["user_id"].numpy().reshape(-1)[offsets].astype(np.int64),
                "item_id": item_dict["item_id"][:, -1].numpy()[offsets].astype(np.int64),
                "seq_len": batch_dict["seq_len"].numpy().reshape(-1)[offsets].astype(np.int64),
                "effective_length": mask.sum(1).numpy()[offsets].astype(np.int16)}
        for label in LABELS:
            truth = batch_dict[label].numpy().reshape(-1)
            if not np.isin(truth, [0, 1]).all():
                raise ValueError(f"Nonbinary sampled label: {label}")
            meta[label] = truth[offsets].astype(np.uint8)
        if np.any(meta["effective_length"] > np.minimum(meta["seq_len"], params["max_len"])):
            raise ValueError("Invalid sampled history length")
        if not np.all(item_dict["action"][:, -1].numpy() == 0):
            raise ValueError("Target action is not masked")
        samples.append({"batch": batch, "meta": meta, "selected_indices": offsets})
        frame.append({"part_id": part, "window_start": 0, "window_rows": window,
                      "selected_rows": count, "formal_retained_rows": coverage[part]})
        engine.atomic_json(output / "progress.json", {"phase": "sampling", "parts_complete": len(samples)})
        del rows, table, payload
        collator.side_cache.clear()
        gc.collect()
    return samples, frame, files


def run_route(params, checkpoint, samples, capture_enabled, output):
    engine.seed_everything(params["seed"])
    model = wrapper.load_model(params, checkpoint, 0)
    capture = Capture(model) if capture_enabled else None
    if capture is not None:
        # PyTorch may reuse the plain graph without invoking newly added hooks.
        # Match the validated compiled diagnostic: reset after registration,
        # then accept capture only if all observed predictions remain bitwise equal.
        torch._dynamo.reset()
    predictions = {label: [] for label in LABELS}
    arrays = []
    try:
        for i, sample in enumerate(samples):
            with torch.no_grad(), model._get_amp_context():
                values = model.forward(tree_map(sample["batch"], lambda tensor: tensor.clone()))
            for label in LABELS:
                predictions[label].append(values[f"{label}_pred"].detach().float().cpu().numpy().reshape(-1)[sample["selected_indices"]])
            if capture is not None:
                arrays.append(capture.consume(sample["selected_indices"]))
            phase = "captured" if capture_enabled else "plain"
            logging.info("%s forward %s/%s complete", phase, i + 1, len(samples))
            engine.atomic_json(output / "progress.json", {"phase": phase, "batches_complete": i + 1})
    finally:
        if capture is not None:
            capture.close()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {label: np.concatenate(parts) for label, parts in predictions.items()}, arrays


def historical_predictions(directory, samples, checkpoint_hash):
    receipt = json.loads((directory / "E/manifest.json").read_text())
    complete = json.loads((directory / "E/complete.json").read_text())
    if receipt["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Historical predictions use a different checkpoint")
    if (receipt.get("model") != "HiFormer" or receipt.get("dataset") != DATASET
            or receipt.get("variant") != "E" or not complete.get("metric_check_passed")
            or complete.get("smoke_only") is not False):
        raise ValueError("Historical predictions are not an accepted formal HiFormer E export")
    result = {label: [] for label in LABELS}
    for sample in samples:
        meta = sample["meta"]
        part = int(meta["part_id"][0])
        path = directory / "E" / f"part-{part:05d}-offset-{0:012d}.parquet"
        engine.verify_shard(path)
        rows = pq.read_table(path).to_pandas().set_index("row_offset", verify_integrity=True).loc[meta["row_offset"]]
        for column, values in meta.items():
            if column == "row_offset":
                continue
            if not np.array_equal(rows[column].to_numpy(), values):
                raise ValueError(f"Historical sample identity mismatch: {column}")
        for label in LABELS:
            result[label].append(rows[f"{label}_pred"].to_numpy(dtype=np.float32))
    return {label: np.concatenate(values) for label, values in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-part", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--prediction-dir", type=Path, default=ROOT / "artifacts/rise_sample_analysis/predictions/hiformer_TencentGR_10M_Action_20260923")
    args = parser.parse_args()
    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise RuntimeError("Requires a single allocated GPU for diagnostic inference")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "artifacts/rise_attention_redistribution"):
        raise ValueError("Output must be in dedicated artifact directory")
    output.mkdir(parents=True, exist_ok=False)
    wrapper.configure_engine(DATASET)
    record = wrapper.reference_record(DATASET)
    log = ROOT / record["checkpoint_log"]
    checkpoint = log.with_suffix(".model")
    params, historical_metrics = engine.parse_log(log)
    expected = {"model": "HiFormer", "dataset_id": DATASET, "seed": 20262028,
                "world_size": 2, "batch_size": 16384, "max_len": 100,
                "enable_bf16": True, "enable_torch_compile": True,
                "epochs": 1, "accumulation_steps": 1, "dataloader_seed": 2027,
                "sisa_enabled": True, "sisa_parameter_seed": 20260822,
                "sisa_score_dim": 16, "sisa_score_scale": 1.0}
    if any(params.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen training configuration changed")
    if params.get("sisa_variant", "full") != "full":
        raise ValueError("Expected full RISE checkpoint")
    checkpoint_identity = engine.file_identity(checkpoint)
    checkpoint_hash = engine.sha256(checkpoint)
    log_identity = engine.file_identity(log)
    log_hash = engine.sha256(log)
    source_hashes = wrapper.source_hashes()
    protocol = ROOT / "experiments/rise_attention_redistribution/protocol.md"
    for file in (Path(__file__).resolve(), LEGACY / "observe_mechanism.py", protocol):
        source_hashes[str(file.relative_to(ROOT))] = engine.sha256(file)
    frozen = output / "frozen_sources"
    for relative in source_hashes:
        target = frozen / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
        if engine.sha256(target) != source_hashes[relative]:
            raise ValueError(f"Source changed while freezing: {relative}")
    metadata_path = Path(engine.find_meta_data_json(params["test_user_info"]))
    metadata_identity = engine.file_identity(metadata_path)
    metadata_hash = engine.sha256(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    frozen_inputs = output / "frozen_inputs"
    frozen_inputs.mkdir()
    for source, name, digest in ((log, "checkpoint.log", log_hash),
                                 (metadata_path, "meta_data.json", metadata_hash)):
        target = frozen_inputs / name
        target.write_bytes(source.read_bytes())
        if engine.sha256(target) != digest:
            raise ValueError(f"Input changed while freezing: {source}")
    manifest = {"schema_version": 1, "model": "HiFormer", "dataset": DATASET, "seed": params["seed"],
                "site": SITE, "head": 0, "sample_seed": args.sample_seed,
                "samples_per_part": args.samples_per_part, "inference_batch_size": params["batch_size"],
                "full_forward_rows_per_route": 8 * params["batch_size"],
                "sampling": "SHA256 priorities within first 16384 retained rows of each of 8 test parts",
                "population_representative": False, "checkpoint": checkpoint_identity,
                "checkpoint_sha256": checkpoint_hash, "reference_record": record,
                "checkpoint_log": log_identity, "checkpoint_log_sha256": log_hash,
                "training_params": params, "historical_metrics": historical_metrics,
                "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
                "source_sha256": source_hashes, "device": device_metadata(),
                "action_vocab": metadata["action_vocab"], "metadata_sha256": metadata_hash,
                "metadata_file": metadata_identity,
                "frozen_input_files": {"frozen_inputs/checkpoint.log": log_hash,
                                       "frozen_inputs/meta_data.json": metadata_hash},
                "reconstruction": "FP32 softmax from actual compiled Q/K/Delta; pre-dropout; native hard mask",
                "same_checkpoint_local_comparison": True, "independent_baseline_comparison": False,
                "strict_capture_atol": 0, "strict_capture_rtol": 0,
                "compiler_cache_reset_after_hook_registration": True,
                "historical_cross_batch_atol": 5e-4, "historical_cross_batch_rtol": 5e-3}
    engine.atomic_json(output / "manifest.json", manifest)
    samples, frame, files = collect_samples(params, args.samples_per_part, args.sample_seed, output)
    save_sample_cache(output / "sample_batches.pt", samples)
    meta = pd.concat([pd.DataFrame(item["meta"]) for item in samples], ignore_index=True)
    meta.insert(0, "sample_index", np.arange(len(meta)))
    if meta.duplicated(["part_id", "row_offset"]).any():
        raise ValueError("Duplicate sampled rows")
    meta.to_csv(output / "meta.csv", index=False)
    manifest.update(sampling_frame=frame, data_files=files, examples=len(meta), unique_users=int(meta.user_id.nunique()))
    engine.atomic_json(output / "manifest.json", manifest)
    plain, _ = run_route(params, checkpoint, samples, False, output)
    historical = historical_predictions(args.prediction_dir, samples, checkpoint_hash)
    historical_check = meta.copy()
    for label in LABELS:
        historical_check[f"{label}_plain"] = plain[label]
        historical_check[f"{label}_historical"] = historical[label]
    historical_check.to_csv(output / "historical_prediction_check.csv", index=False)
    historical_gate = prediction_comparison(plain, historical, atol=5e-4, rtol=5e-3)
    engine.atomic_json(output / "historical_prediction_gate.json", historical_gate)
    if not historical_gate["passed"]:
        raise RuntimeError("Diagnostic predictions failed historical cross-batch gate")
    captured, pieces = run_route(params, checkpoint, samples, True, output)
    gate = prediction_comparison(captured, plain)
    predictions = meta.copy()
    for label in LABELS:
        for name, values in (("plain", plain), ("captured", captured), ("historical", historical)):
            predictions[f"{label}_{name}"] = values[label]
    predictions.to_csv(output / "prediction_check.csv", index=False)
    engine.atomic_json(output / "prediction_gate.json", gate)
    if not gate["bitwise_equal"]:
        raise RuntimeError("Capture changed predictions; reject attention evidence")
    raw = {name: np.concatenate([piece[name] for piece in pieces]) for name in pieces[0]}
    raw.update(valid_mask=np.concatenate([sample["batch"][2].numpy()[sample["selected_indices"]].astype(bool) for sample in samples]),
               history_item_ids=np.concatenate([sample["batch"][1]["item_id"][:, :-1].numpy()[sample["selected_indices"]] for sample in samples]),
               history_actions=np.concatenate([sample["batch"][1]["action"][:, :-1].numpy()[sample["selected_indices"]] for sample in samples]),
               target_item_ids=meta.item_id.to_numpy())
    if raw["native_scores"].shape != (len(meta), 1, 100):
        raise ValueError("Unexpected target attention shape")
    np.savez_compressed(output / "raw_attention.npz", **raw)
    rows = []
    for i in range(len(meta)):
        valid = raw["valid_mask"][i]
        nonempty = bool(valid.any())
        rows.append({"sample_index": i, "head": 0, "tv": float(raw["tv"][i, 0]),
                     "omega": float(raw["omega"][i, 0]),
                     "top_before": int(np.argmax(raw["weight_before"][i, 0])) if nonempty else -1,
                     "top_after": int(np.argmax(raw["weight_after"][i, 0])) if nonempty else -1})
    stats = pd.DataFrame(rows)
    stats.to_csv(output / "stats.csv", index=False)
    if engine.file_identity(checkpoint) != checkpoint_identity or engine.sha256(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed during export")
    if engine.file_identity(log) != log_identity or engine.sha256(log) != log_hash:
        raise ValueError("Checkpoint log changed during export")
    if engine.file_identity(metadata_path) != metadata_identity or engine.sha256(metadata_path) != metadata_hash:
        raise ValueError("Dataset metadata changed during export")
    if any(engine.sha256(ROOT / path) != digest for path, digest in source_hashes.items()):
        raise ValueError("Source changed during export")
    if any(engine.file_identity(Path(path)) != value for path, value in files.items()):
        raise ValueError("Data file identity changed during export")
    valid = meta.effective_length > 0
    summary = {"examples": len(meta), "nonempty_examples": int(valid.sum()), "empty_examples": int((~valid).sum()),
               "unique_users": int(meta.user_id.nunique()), "tv_mean": float(stats.loc[valid, "tv"].mean()),
               "tv_median": float(stats.loc[valid, "tv"].median()),
               "tv_p10": float(stats.loc[valid, "tv"].quantile(.1)), "tv_p90": float(stats.loc[valid, "tv"].quantile(.9)),
               "top_key_changed_fraction": float((stats.loc[valid, "top_before"] != stats.loc[valid, "top_after"]).mean()),
               "prediction_bitwise_equal": gate["bitwise_equal"], "historical_cross_batch_gate_passed": historical_gate["passed"]}
    engine.atomic_json(output / "summary.json", summary)
    engine.atomic_json(output / "complete.json", {"completed": True, "summary": summary,
        "files_sha256": {str(path.relative_to(output)): engine.sha256(path) for path in sorted(output.iterdir()) if path.is_file()}})
    logging.info("Completed: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
