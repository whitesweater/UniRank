#!/usr/bin/env python3
"""Repackage the frozen 512 actual examples into nested batches 64/256/512.

No new examples, padding, repeat sampling, or prediction-based selection. The
64-row entry exactly preserves the first input used by the old benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from observe_mechanism import (
    ROOT, atomic_json, file_identity, identity_hash, load_sample_cache,
    save_sample_cache, sha256, subset_batch,
)

SIZES = (64, 256, 512)


def tensor_tree_hash(batch):
    digest = hashlib.sha256()
    def visit(node, path):
        if torch.is_tensor(node):
            header = {"path": path, "dtype": str(node.dtype), "shape": list(node.shape)}
            digest.update(json.dumps(header, sort_keys=True).encode() + b"\0")
            digest.update(node.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(node, dict):
            for key in sorted(node):
                visit(node[key], path + [str(key)])
        elif isinstance(node, (tuple, list)):
            digest.update(type(node).__name__.encode() + b"\0")
            for index, value in enumerate(node):
                visit(value, path + [str(index)])
        else:
            raise TypeError(f"Unsupported tensor tree node: {type(node)}")
    visit(batch, [])
    return digest.hexdigest()


def concatenate_trees(trees, counts):
    first = trees[0]
    if torch.is_tensor(first):
        if any(not torch.is_tensor(value) or value.shape[0] != count
               or value.shape[1:] != first.shape[1:] or value.dtype != first.dtype
               or value.device.type != "cpu" for value, count in zip(trees, counts)):
            raise ValueError("Batch tensor dimensions, dtype or CPU placement differ")
        return torch.cat(trees, dim=0).contiguous()
    if isinstance(first, dict):
        if any(not isinstance(value, dict) or set(value) != set(first) for value in trees):
            raise ValueError("Batch dictionary structure differs")
        return {key: concatenate_trees([value[key] for value in trees], counts) for key in first}
    if isinstance(first, (tuple, list)):
        if any(type(value) is not type(first) or len(value) != len(first) for value in trees):
            raise ValueError("Batch sequence structure differs")
        return type(first)(concatenate_trees([value[i] for value in trees], counts)
                           for i in range(len(first)))
    raise TypeError(f"Unsupported batch node: {type(first)}")


def build_entries(samples, sizes=SIZES):
    if not samples or not sizes or len(set(sizes)) != len(sizes) or any(size < 1 for size in sizes):
        raise ValueError("Nonempty samples and distinct positive batch sizes required")
    columns = set(samples[0]["meta"])
    counts = [len(sample["meta"]["row_offset"]) for sample in samples]
    for sample, count in zip(samples, counts):
        if set(sample["meta"]) != columns or any(len(value) != count for value in sample["meta"].values()):
            raise ValueError("Inconsistent source metadata schema or row counts")
        if any(value.ndim != 1 or value.dtype != samples[0]["meta"][key].dtype
               for key, value in sample["meta"].items()):
            raise ValueError("Metadata dtype or dimensionality drift would change row identities")
    merged_meta = {key: np.concatenate([sample["meta"][key] for sample in samples])
                   for key in samples[0]["meta"]}
    identities = list(zip(merged_meta["part_id"].tolist(), merged_meta["row_offset"].tolist()))
    if len(identities) != len(set(identities)):
        raise ValueError("Source sample cache contains repeated row IDs")
    if max(sizes) > len(identities):
        raise ValueError("Requested batch exceeds actual unique source rows")
    merged_batch = concatenate_trees([sample["batch"] for sample in samples], counts)
    entries = [{"batch": subset_batch(merged_batch, np.arange(size)),
                "meta": {key: values[:size].copy() for key, values in merged_meta.items()}}
               for size in sizes]
    return entries, merged_meta


def source_contract(source_dir):
    manifest_path = source_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_cache = source_dir / "sample_batches.pt"
    actual = sha256(source_cache)
    if actual != manifest["sample_cache_sha256"]:
        raise ValueError("Frozen source cache hash changed")
    samples = load_sample_cache(source_cache)
    entries, merged_meta = build_entries(samples)
    if len(merged_meta["row_offset"]) != 512 or manifest["rows"] != 512:
        raise ValueError("This study requires the exact frozen 512 source examples")
    if (len(samples[0]["meta"]["row_offset"]) != 64
            or identity_hash(entries[0]["meta"]) != identity_hash(samples[0]["meta"])
            or tensor_tree_hash(entries[0]["batch"]) != tensor_tree_hash(samples[0]["batch"])):
        raise ValueError("The 64-row benchmark entry must exactly preserve the original first batch")
    rows = pd.read_parquet(source_dir / "sample_rows.parquet")
    if set(rows) != set(merged_meta) or any(not np.array_equal(rows[key].to_numpy(), values)
                                          for key, values in merged_meta.items()):
        raise ValueError("Source row table and source tensor-cache metadata differ")
    return entries, merged_meta, {
        "source_dir": str(source_dir.resolve()), "source_cache_sha256": actual,
        "source_cache_file": file_identity(source_cache),
        "source_manifest_sha256": sha256(manifest_path),
        "source_rows_sha256": sha256(source_dir / "sample_rows.parquet"),
        "source_identity_sha256": identity_hash(merged_meta),
        "source_entry_rows": [len(sample["meta"]["row_offset"]) for sample in samples],
    }


def entry_record(index, sample):
    meta = sample["meta"]
    parts, counts = np.unique(meta["part_id"], return_counts=True)
    return {"sample_index": index, "rows": len(meta["row_offset"]),
            "identity_sha256": identity_hash(meta), "tensor_tree_sha256": tensor_tree_hash(sample["batch"]),
            "part_counts": {str(int(part)): int(count) for part, count in zip(parts, counts)},
            "source_prefix_rows": len(meta["row_offset"]), "unique_row_ids": True}


def verify_cache(directory, source_dir):
    expected, _, contract = source_contract(source_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["source_contract"] != contract:
        raise ValueError("Cache source contract differs")
    cache_path = directory / "benchmark_batches.pt"
    if sha256(cache_path) != manifest["benchmark_cache_sha256"]:
        raise ValueError("Benchmark cache bytes changed")
    actual = load_sample_cache(cache_path)
    if len(actual) != len(SIZES) or len(manifest["entries"]) != len(SIZES):
        raise ValueError("Expected exactly three benchmark entries")
    for i, (left, right) in enumerate(zip(expected, actual)):
        record = entry_record(i, right)
        if record != entry_record(i, left) or record != manifest["entries"][i]:
            raise ValueError(f"Entry {i} identity or tensor content changed")
    row_table = pd.read_parquet(directory / "entry_rows.parquet")
    for index, sample in enumerate(actual):
        rows = row_table.loc[row_table.sample_index == index].drop(columns=["sample_index", "entry_row"])
        if len(rows) != SIZES[index] or any(not np.array_equal(rows[key].to_numpy(), value)
                                          for key, value in sample["meta"].items()):
            raise ValueError(f"Entry {index} row-table identity changed")
    if sha256(directory / "entry_rows.parquet") != manifest["entry_rows_sha256"]:
        raise ValueError("Entry row-table bytes changed")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "artifacts/rise_sample_analysis/mechanism"):
        parser.error("Output must be under artifacts/rise_sample_analysis/mechanism")
    if args.verify_only:
        manifest = verify_cache(output, source)
        print(json.dumps({"verified": True, "entries": manifest["entries"]}, indent=2))
        return
    entries, merged_meta, contract = source_contract(source)
    output.mkdir(parents=True, exist_ok=False)
    save_sample_cache(output / "benchmark_batches.pt", entries)
    tables = []
    for i, sample in enumerate(entries):
        table = pd.DataFrame(sample["meta"])
        table.insert(0, "entry_row", np.arange(len(table)))
        table.insert(0, "sample_index", i)
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_parquet(output / "entry_rows.parquet", index=False)
    manifest = {
        "schema_version": 1, "source_contract": contract,
        "source_unique_rows": len(merged_meta["row_offset"]),
        "sizes": list(SIZES), "entries": [entry_record(i, sample) for i, sample in enumerate(entries)],
        "benchmark_cache_sha256": sha256(output / "benchmark_batches.pt"),
        "entry_rows_sha256": sha256(output / "entry_rows.parquet"),
        "builder_sha256": sha256(Path(__file__)),
        "benchmark_script_sha256": sha256(Path(__file__).with_name("benchmark_inference.py")),
        "construction": "concatenate original cache entries in original order, then take nested prefixes",
        "no_repeated_rows_within_entry": True, "nested_overlap_between_entries_is_intentional": True,
        "entry_zero_identical_to_old_first_64_input": True,
        "sampling_scope": "same fixed diagnostic sample; batch composition changes across prefix sizes",
    }
    atomic_json(output / "manifest.json", manifest)
    verify_cache(output, source)
    atomic_json(output / "complete.json", {"verified": True, "source_unique_rows": 512,
                                           "sizes": list(SIZES), "entries": manifest["entries"]})
    print(json.dumps({"verified": True, "output": str(output), "entries": manifest["entries"]}, indent=2))


if __name__ == "__main__":
    main()
