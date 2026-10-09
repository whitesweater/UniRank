"""Read-only CPU check that the exact 64-row native pooling test is nonvacuous."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(plan_path):
    plan = json.loads(plan_path.read_text())
    sample_path = Path(plan["historical_attention_source"]) / "sample_batches.pt"
    sample_sha = sha(sample_path)
    if sample_sha != plan["frozen_files_sha256"][str(sample_path)]:
        raise ValueError("Cached sample SHA differs from frozen runner plan")
    samples = torch.load(sample_path, map_location="cpu", weights_only=True, mmap=True)
    batch = samples[0]["batch"]
    mask = batch[2][:64].numpy().astype(bool)
    attributes = {key: value[:64].numpy() for key, value in batch[1].items()}
    if mask.shape != (64, 100):
        raise ValueError(mask.shape)
    results = {}
    for route in ("reverse", "random"):
        mapping = np.tile(np.arange(100), (64, 1))
        rng = np.random.default_rng(plan["permutation_seed"])
        for row in range(64):
            valid = np.flatnonzero(mask[row])
            mapping[row, valid] = valid[::-1] if route == "reverse" else rng.permutation(valid)
        full_mapping = np.column_stack([mapping, np.full(64, 100)])
        event_changed = np.zeros((64, 100), dtype=bool)
        attribute_details = {}
        for key, original in attributes.items():
            if original.shape[:2] != (64, 101):
                raise ValueError((key, original.shape))
            index = full_mapping.reshape((64, 101) + (1,) * (original.ndim - 2))
            changed = np.take_along_axis(original, index, axis=1)
            equal = changed == original
            if np.issubdtype(original.dtype, np.floating):
                equal |= np.isnan(changed) & np.isnan(original)
            if not equal[:, -1].all():
                raise ValueError("Candidate changed")
            different = ~equal[:, :100]
            position_changed = different if different.ndim == 2 else np.any(different, axis=tuple(range(2, different.ndim)))
            if np.any(position_changed & ~mask):
                raise ValueError("Padding content changed")
            event_changed |= position_changed
            attribute_details[key] = {"changed_scalar_entries": int(different.sum()),
                "changed_history_positions": int(position_changed.sum()),
                "rows_with_changed_history": int(position_changed.any(axis=1).sum()),
                "candidate_exactly_fixed": True}
        results[route] = {"raw_event_positions_changed_any_attribute": int(event_changed.sum()),
            "rows_with_raw_history_content_change": int(event_changed.any(axis=1).sum()),
            "eligible_rows_with_no_raw_content_change": np.flatnonzero((mask.sum(axis=1) >= 2) & ~event_changed.any(axis=1)).tolist(),
            "valid_index_positions_moved": int((mapping != np.arange(100)[None, :]).sum()),
            "candidate_fixed_all_attributes": True, "padding_fixed_all_attributes": True,
            "per_attribute": attribute_details}
    return {"case": plan["case"], "window_index": 0, "rows": 64,
        "eligible_rows": int((mask.sum(axis=1) >= 2).sum()),
        "history_lengths": mask.sum(axis=1).astype(int).tolist(),
        "source_sample_sha256": sample_sha, "runner_plan_sha256": sha(plan_path),
        "no_training": True, "cpu_only": True, "routes": results,
        "interpretation": "Counts establish whether the existing 64-row numerical pooling check actually changes raw historical content; they do not quantify model performance."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    record = {"script_sha256": sha(__file__), "checks": [audit(path) for path in args.plans]}
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "checks": [{"case": v["case"], "eligible_rows": v["eligible_rows"], "routes": {route: {k: result[k] for k in ("raw_event_positions_changed_any_attribute", "rows_with_raw_history_content_change", "valid_index_positions_moved")} for route, result in v["routes"].items()}} for v in record["checks"]]}, indent=2))
