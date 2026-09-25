"""Frozen RankMixer/KuaiRand attention and local score-bias sensitivity; no training."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

import export_rise_attention_redistribution as frozen_diagnostic
from observe_mechanism import device_metadata, tree_map
from unirank.pytorch.layers.attentions.dot_product_attention import ScaledDotProductAttention

engine = frozen_diagnostic.engine
selected_offsets = frozen_diagnostic.selected_offsets
reconstruct_attention = frozen_diagnostic.reconstruct_attention
save_sample_cache = frozen_diagnostic.save_sample_cache
DATASET = "KuaiRand_Video_Action"
MODEL = "RankMixer"
LABELS = ("is_click", "is_follow", "is_like", "is_comment", "is_forward", "long_view")
SITE = "attention_layers.dot_attention"
REFERENCE = ROOT / "experiments/sisa_single_seed20262028/results/runs.csv"
FORMAL_RETAINED_ROWS = 32_964_608
BCE_EPSILON = 1e-7


def reference_record():
    with REFERENCE.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["model"] == MODEL and row["dataset"] == DATASET
                and row["setting"] == "sisa" and row["seed"] == "20262028"]
    if len(rows) != 1 or any(rows[0][key] != "True" for key in
                            ("complete", "protocol_valid", "completion_marker", "finite_test_metrics")):
        raise ValueError("Expected one accepted complete RankMixer KuaiRand full-RISE record")
    return rows[0]


def prediction_comparison(actual, reference):
    if set(actual) != set(LABELS) or set(reference) != set(LABELS):
        raise ValueError("Prediction gate requires exactly all six KuaiRand labels")
    result = {}
    for label in LABELS:
        a, b = np.asarray(actual[label]), np.asarray(reference[label])
        if not a.size or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Invalid prediction shape or nonfinite value")
        result[label] = {"bitwise_equal": bool(np.array_equal(a, b)),
                         "max_abs_error": float(np.max(np.abs(a - b)))}
    equal = all(value["bitwise_equal"] for value in result.values())
    return {"atol": 0, "rtol": 0, "labels": result, "bitwise_equal": equal, "passed": equal}


def binary_cross_entropy(truth, probability):
    truth, probability = np.asarray(truth), np.asarray(probability, dtype=np.float64)
    if (truth.shape != probability.shape or not np.isin(truth, [0, 1]).all()
            or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))):
        raise ValueError("BCE requires aligned binary labels and finite probabilities in [0,1]")
    p = np.clip(probability, BCE_EPSILON, 1 - BCE_EPSILON)
    return -(truth * np.log(p) + (1 - truth) * np.log1p(-p))


def site_module(model):
    matches = [(name, module) for name, module in model.named_modules()
               if isinstance(module, ScaledDotProductAttention)
               and ".".join(part for part in name.split(".") if part != "_orig_mod") == SITE]
    if len(matches) != 1 or matches[0][1].attention_activation_type != "SoftMax":
        raise ValueError(f"Expected one target softmax site, found {len(matches)}")
    return matches[0][1]


def zero_score_bias(args, kwargs):
    """Keep Q/K/V, native mask, scale and tensor-bias kernel path unchanged."""
    bias = kwargs.get("score_bias")
    if bias is None or len(args) != 3:
        raise ValueError("Expected native target site's three Q/K/V args and tensor score_bias")
    return args, dict(kwargs, score_bias=torch.zeros_like(bias))


class Capture:
    def __init__(self, model):
        self.entries = []
        self.site_info = None
        self.qkv_signatures = []
        self.handle = site_module(model).register_forward_pre_hook(self.hook, with_kwargs=True)

    def hook(self, module, args, kwargs):
        self.entries.append((args, kwargs))

    def consume(self, indices):
        if len(self.entries) != 1:
            raise ValueError(f"Expected one native-site call, got {len(self.entries)}")
        args, kwargs = self.entries.pop()
        query, key = args[:2]
        info = {"site": SITE, "num_heads": query.shape[1], "head_dim": query.shape[-1],
                "queries": query.shape[-2], "keys": key.shape[-2], "scale_divisor": kwargs.get("scale"),
                "query_dtype": str(query.dtype), "key_dtype": str(key.dtype),
                "bias_dtype": str(kwargs["score_bias"].dtype), "mask_dtype": str(kwargs["mask"].dtype)}
        if self.site_info is not None and self.site_info != info:
            raise ValueError("Attention structure or dtype changed across sampling windows")
        self.site_info = info
        def take(tensor):
            return tensor.index_select(0, torch.as_tensor(indices, device=tensor.device))
        selected_qkv = [take(tensor) for tensor in args[:3]]
        self.qkv_signatures.append({name: {
            "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()}
            for name, tensor in zip(("query", "key", "value"), selected_qkv)})
        mask = take(kwargs["mask"])
        raw = reconstruct_attention(selected_qkv[0], selected_qkv[1], take(kwargs["score_bias"]), mask, kwargs.get("scale"))
        raw["native_mask"] = mask.squeeze(-2).detach().bool().cpu().numpy()
        return raw

    def close(self):
        self.handle.remove()
        self.entries.clear()


class LocalOff(Capture):
    def hook(self, module, args, kwargs):
        # Observe the exact upstream tensors before replacing only the bias.
        self.entries.append((args, kwargs))
        return zero_score_bias(args, kwargs)


def load_model(params, checkpoint, device):
    from model_zoo.RankMixer import RankMixer
    configured = dict(params, gpu=device)
    feature_map = engine.feature_map_from_config(configured)
    if tuple(feature_map.labels) != LABELS:
        raise ValueError("Historical label order differs from six-label KuaiRand protocol")
    model = RankMixer(feature_map, **configured)
    if not model._torch_compile_enabled:
        raise RuntimeError("Historical checkpoint requires successful compile wrapping")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    model.optimizers = []
    model.optimizer = model.dense_optimizer = model.sparse_optimizer = None
    model.eval()
    return model


def collect_samples(params, count, seed, output):
    feature_map = engine.feature_map_from_config(params)
    if tuple(feature_map.labels) != LABELS:
        raise ValueError("Unexpected label mapping")
    dataset = engine.IdentifiedDataset(
        params["test_data"], params["test_user_info"], params["test_item_info"],
        batch_size=params["batch_size"], shuffle=False, distributed=True,
        rank=0, world_size=2, drop_last=True, seed=params["dataloader_seed"])
    coverage_by_rank = frozen_diagnostic.wrapper.coverage_by_rank(dataset, 2)
    coverage = {}
    for rank in coverage_by_rank:
        if set(rank) & set(coverage):
            raise ValueError("Duplicate physical part assignment")
        coverage.update(rank)
    if len(coverage) != 8 or sum(coverage.values()) != FORMAL_RETAINED_ROWS:
        raise ValueError("Formal KuaiRand drop-last coverage changed")
    collator = engine.BlockedBatchCollator(feature_map, params["max_len"], dataset.column_index,
                                         params["test_user_info"], params["test_item_info"], cache_size=1)
    samples, frame, files = [], [], {}
    for block in sorted(dataset.blocks, key=lambda b: int(b["part_id"])):
        part, window = int(block["part_id"]), params["batch_size"]
        if coverage[part] < window:
            raise ValueError("Sampling window exceeds retained formal coverage")
        for field in ("data_file", "user_info_file", "item_info_file"):
            files[str(block[field])] = engine.file_identity(Path(block[field]))
        logging.info("Sampling physical part %s", part)
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


def run_route(params, checkpoint, samples, route, output):
    if route not in ("plain", "capture", "local_off"):
        raise ValueError("Unknown inference route")
    engine.seed_everything(params["seed"])
    model = load_model(params, checkpoint, 0)
    observer = Capture(model) if route == "capture" else LocalOff(model) if route == "local_off" else None
    if observer is not None:
        torch._dynamo.reset()
    predictions, arrays = {label: [] for label in LABELS}, []
    digests = {label: hashlib.sha256() for label in LABELS}
    full_rows = 0
    site_info = None
    try:
        for batch_index, sample in enumerate(samples):
            with torch.no_grad(), model._get_amp_context():
                values = model.forward(tree_map(sample["batch"], lambda tensor: tensor.clone()))
            batch_size = None
            for label in LABELS:
                whole = values[f"{label}_pred"].detach().float().cpu().numpy().reshape(-1)
                if not np.isfinite(whole).all():
                    raise ValueError("Nonfinite route prediction")
                if batch_size is not None and batch_size != len(whole):
                    raise ValueError("Prediction batch size differs across labels")
                batch_size = len(whole)
                digests[label].update(whole.tobytes())
                predictions[label].append(whole[sample["selected_indices"]])
            full_rows += batch_size
            if route == "capture":
                arrays.append(observer.consume(sample["selected_indices"]))
                site_info = observer.site_info
            elif route == "local_off":
                observer.consume(sample["selected_indices"])
                site_info = observer.site_info
            logging.info("%s complete %s/%s full windows", route, batch_index + 1, len(samples))
            engine.atomic_json(output / "progress.json", {"phase": route, "batches_complete": batch_index + 1})
    finally:
        if observer is not None:
            observer.close()
    del model, values
    gc.collect()
    torch.cuda.empty_cache()
    return ({label: np.concatenate(parts) for label, parts in predictions.items()}, arrays,
            {"route": route, "full_forward_rows": full_rows,
             "full_prediction_sha256": {label: value.hexdigest() for label, value in digests.items()},
             "site_info": site_info, "intervened_site": SITE if route == "local_off" else None,
             "selected_qkv_signatures_by_batch": observer.qkv_signatures if observer is not None else None})


def attach_losses(meta, plain, captured, off):
    predictions = meta.copy()
    for label in LABELS:
        for name, values in (("plain", plain), ("capture", captured), ("local_off", off)):
            predictions[f"{label}_{name}"] = values[label]
        for name, values in (("plain", plain), ("local_off", off)):
            predictions[f"{label}_bce_{name}"] = binary_cross_entropy(meta[label].to_numpy(), values[label])
        predictions[f"{label}_bce_gain"] = predictions[f"{label}_bce_local_off"] - predictions[f"{label}_bce_plain"]
    rows = []
    for group, selected in (("all", np.ones(len(meta), dtype=bool)),
                            ("nonempty_history", meta.effective_length.to_numpy() > 0),
                            ("empty_history", meta.effective_length.to_numpy() == 0)):
        for label in LABELS:
            gain = predictions.loc[selected, f"{label}_bce_gain"]
            rows.append({"group": group, "label": label, "examples": int(selected.sum()),
                         "positives": int(meta.loc[selected, label].sum()),
                         "improved": int((gain > 0).sum()), "degraded": int((gain < 0).sum()),
                         "unchanged": int((gain == 0).sum()),
                         "mean_bce_plain": float(predictions.loc[selected, f"{label}_bce_plain"].mean()) if selected.any() else None,
                         "mean_bce_local_off": float(predictions.loc[selected, f"{label}_bce_local_off"].mean()) if selected.any() else None,
                         "mean_bce_gain": float(gain.mean()) if selected.any() else None})
    return predictions, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise RuntimeError("Requires one allocated GPU for fixed-checkpoint diagnostic inference")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "artifacts/rise_attention_semantic"):
        raise ValueError("Output must be in dedicated semantic attention artifact directory")
    output.mkdir(parents=True, exist_ok=False)
    overrides = {"DATASET": {"original": engine.DATASET, "effective": DATASET},
                 "LABELS": {"original": list(engine.LABELS), "effective": list(LABELS)}}
    engine.DATASET, engine.LABELS = DATASET, LABELS
    record = reference_record()
    log = ROOT / record["checkpoint_log"]
    checkpoint = log.with_suffix(".model")
    params, metrics = engine.parse_log(log)
    expected = {"model": MODEL, "dataset_id": DATASET, "seed": 20262028,
                "world_size": 2, "batch_size": 16384, "max_len": 100,
                "enable_bf16": True, "enable_torch_compile": True,
                "epochs": 1, "accumulation_steps": 1, "dataloader_seed": 2027,
                "sisa_enabled": True, "sisa_parameter_seed": 20260822,
                "sisa_score_dim": 16, "sisa_score_scale": 1.0}
    if any(params.get(key) != value for key, value in expected.items()) or params.get("sisa_variant", "full") != "full":
        raise ValueError("Frozen accepted training configuration changed")
    if metrics != json.loads(record["metrics_json"]):
        raise ValueError("Final checkpoint log metrics disagree with accepted run record")
    metadata_path = Path(engine.find_meta_data_json(params["test_user_info"]))
    watched = {"checkpoint": checkpoint, "checkpoint_log": log, "metadata": metadata_path, "reference_csv": REFERENCE}
    identities = {name: engine.file_identity(path) for name, path in watched.items()}
    hashes = {name: engine.sha256(path) for name, path in watched.items()}
    metadata = json.loads(metadata_path.read_text())
    sources = frozen_diagnostic.wrapper.source_hashes()
    for path in (Path(__file__), ROOT / "scripts/export_rise_attention_redistribution.py",
                 ROOT / "scripts/submit_rise_semantic_attention.sbatch",
                 ROOT / "experiments/rise_sample_analysis/scripts/observe_mechanism.py",
                 ROOT / "experiments/rise_attention_semantic/protocol.md"):
        sources[str(path.relative_to(ROOT))] = engine.sha256(path)
    for relative, digest in sources.items():
        target = output / "frozen_sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
        if engine.sha256(target) != digest:
            raise ValueError(f"Source changed while freezing: {relative}")
    frozen = output / "frozen_inputs"
    frozen.mkdir()
    for name in ("checkpoint_log", "metadata", "reference_csv"):
        target = frozen / watched[name].name
        target.write_bytes(watched[name].read_bytes())
        if engine.sha256(target) != hashes[name]:
            raise ValueError(f"Input changed while freezing: {name}")
    engine.atomic_json(frozen / "params.json", params)
    manifest = {"schema_version": 1, "model": MODEL, "dataset": DATASET, "seed": params["seed"],
                "variant": "E", "labels": list(LABELS), "site": SITE,
                "sample_seed": 20260923, "samples_per_part": 64, "physical_test_parts": 8,
                "inference_batch_size": 16384, "full_forward_rows_per_route": 131072,
                "sampling": "SHA256(seed:part:row) priorities in first retained 16384 rows per physical test part; choose 64 sorted offsets",
                "population_representative": False, "reference_record": record,
                "training_params": params, "historical_metrics": metrics,
                "input_identities": identities, "input_sha256": hashes,
                "checkpoint_sha256": hashes["checkpoint"], "checkpoint": identities["checkpoint"],
                "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
                "source_sha256": sources, "device": device_metadata(), "engine_constant_overrides": overrides,
                "action_vocab": metadata["action_vocab"], "torch_version": torch.__version__,
                "reconstruction": "FP32 pre-dropout softmax from actual compiled Q/K and dtype-cast bias; native allowed set; empty histories undefined",
                "local_off": "At attention_layers.dot_attention only, replace score_bias by zeros_like; preserve tensor-bias SDPA path, QKV, mask, other model logic",
                "same_checkpoint_local_sensitivity": True, "independent_baseline_comparison": False,
                "bce_gain_definition": "BCE(local_off,y) - BCE(plain_full_RISE,y); positive means lower BCE with local correction",
                "bce_probability_clip_epsilon": BCE_EPSILON,
                "historical_rowwise_prediction_check": "not_available_not_claimed",
                "strict_capture_atol": 0, "strict_capture_rtol": 0,
                "compiler_cache_reset_after_hook_registration": True}
    engine.atomic_json(output / "manifest.json", manifest)
    samples, frame, files = collect_samples(params, 64, 20260923, output)
    save_sample_cache(output / "sample_batches.pt", samples)
    meta = pd.concat([pd.DataFrame(sample["meta"]) for sample in samples], ignore_index=True)
    meta.insert(0, "sample_index", np.arange(len(meta)))
    if len(meta) != 512 or meta.duplicated(["part_id", "row_offset"]).any():
        raise ValueError("Unexpected sample count or duplicate sample identity")
    meta.to_csv(output / "meta.csv", index=False)
    manifest.update(sampling_frame=frame, data_files=files, examples=len(meta), unique_users=int(meta.user_id.nunique()))
    engine.atomic_json(output / "manifest.json", manifest)
    plain, _, plain_info = run_route(params, checkpoint, samples, "plain", output)
    captured, pieces, capture_info = run_route(params, checkpoint, samples, "capture", output)
    gate = prediction_comparison(captured, plain)
    gate["full_forward_predictions_bitwise_equal"] = plain_info["full_prediction_sha256"] == capture_info["full_prediction_sha256"]
    gate["full_forward_rows"] = plain_info["full_forward_rows"]
    gate["passed"] = gate["bitwise_equal"] and gate["full_forward_predictions_bitwise_equal"]
    engine.atomic_json(output / "prediction_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError("Capture changed predictions; reject attention evidence")
    raw = {name: np.concatenate([piece[name] for piece in pieces]) for name in pieces[0]}
    raw["valid_mask"] = np.concatenate([sample["batch"][2].numpy()[sample["selected_indices"]].astype(bool) for sample in samples])
    for field, plural in (("item_id", "item_ids"), ("action", "actions"), ("duration_bucket", "duration_bucket"),
                          ("primary_tag", "primary_tag"), ("video_type", "video_type"), ("music_type", "music_type")):
        values = np.concatenate([sample["batch"][1][field].numpy()[sample["selected_indices"]] for sample in samples])
        raw[f"history_{plural}"] = values[:, :-1]
        raw[f"target_{plural}"] = values[:, -1]
    if raw["native_scores"].shape != (512, capture_info["site_info"]["num_heads"], 100):
        raise ValueError("Unexpected native attention shape")
    if not np.array_equal(raw["native_mask"], np.broadcast_to(raw["valid_mask"][:, None], raw["native_mask"].shape)):
        raise ValueError("Captured native mask differs from processed history mask")
    np.savez_compressed(output / "raw_attention.npz", **raw)
    off, _, off_info = run_route(params, checkpoint, samples, "local_off", output)
    intervention_gate = {
        "selected_qkv_bitwise_equal": capture_info["selected_qkv_signatures_by_batch"] == off_info["selected_qkv_signatures_by_batch"],
        "site_structure_equal": capture_info["site_info"] == off_info["site_info"],
        "scope": "All 512 selected interactions, all actual heads; Q/K/V dtype, shape and raw bytes"}
    intervention_gate["passed"] = intervention_gate["selected_qkv_bitwise_equal"] and intervention_gate["site_structure_equal"]
    engine.atomic_json(output / "local_intervention_gate.json", intervention_gate)
    if not intervention_gate["passed"]:
        raise RuntimeError("Local intervention changed upstream Q/K/V; reject paired local-sensitivity evidence")
    predictions, loss_summary = attach_losses(meta, plain, captured, off)
    predictions.to_csv(output / "predictions.csv", index=False)
    loss_summary.to_csv(output / "label_sensitivity.csv", index=False)
    stats = []
    for i in range(len(meta)):
        for head in range(raw["native_scores"].shape[1]):
            nonempty = bool(raw["valid_mask"][i].any())
            stats.append({"sample_index": i, "head": head, "tv": float(raw["tv"][i, head]),
                          "omega": float(raw["omega"][i, head]),
                          "top_before": int(np.argmax(raw["weight_before"][i, head])) if nonempty else -1,
                          "top_after": int(np.argmax(raw["weight_after"][i, head])) if nonempty else -1})
    pd.DataFrame(stats).to_csv(output / "stats.csv", index=False)
    for name, path in watched.items():
        if engine.file_identity(path) != identities[name] or engine.sha256(path) != hashes[name]:
            raise ValueError(f"Frozen input changed during export: {name}")
    if any(engine.sha256(ROOT / path) != digest for path, digest in sources.items()):
        raise ValueError("Source changed during export")
    if any(engine.file_identity(Path(path)) != identity for path, identity in files.items()):
        raise ValueError("Data file identity changed during export")
    manifest.update(route_info={"plain": plain_info, "capture": capture_info, "local_off": off_info},
                    site_info=capture_info["site_info"], input_end_sha256=hashes)
    engine.atomic_json(output / "manifest.json", manifest)
    valid = meta.effective_length > 0
    summary = {"examples": len(meta), "unique_users": int(meta.user_id.nunique()),
               "nonempty_examples": int(valid.sum()), "empty_examples": int((~valid).sum()),
               "tv_mean": float(np.nanmean(raw["tv"])), "tv_median": float(np.nanmedian(raw["tv"])),
               "capture_bitwise_equal": gate["passed"], "formal_test_metrics_recomputed": False,
               "interpretation": "Descriptive local sensitivity on fixed diagnostic interactions; no population gain or causal content attribution claim"}
    engine.atomic_json(output / "summary.json", summary)
    engine.atomic_json(output / "complete.json", {"completed": True, "summary": summary,
        "files_sha256": {str(path.relative_to(output)): engine.sha256(path) for path in sorted(output.iterdir()) if path.is_file()}})
    logging.info("Completed: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
