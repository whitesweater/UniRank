from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_benchmark_cache import build_entries, tensor_tree_hash
from observe_mechanism import subset_batch


class BenchmarkCacheTest(unittest.TestCase):
    def samples(self):
        samples = []
        for part in range(3):
            start = part * 2
            samples.append({"meta": {"part_id": np.full(2, part, dtype=np.int32),
                "row_offset": np.array([8, 19], dtype=np.int64),
                "user_id": np.arange(start, start + 2) + 100,
                "Like": np.array([0, 1], dtype=np.uint8)},
                "batch": ({"user_id": torch.arange(start, start + 2) + 100},
                          {"item_id": torch.arange(start * 5, (start + 2) * 5).view(2, 5)},
                          torch.full((2, 4), float(part)), [torch.zeros(2, 4)])})
        return samples

    def test_prefix_entries_preserve_every_real_tensor_and_unique_row_identity(self):
        source = self.samples()
        entries, meta = build_entries(source, sizes=(2, 4, 6))
        self.assertEqual(tensor_tree_hash(entries[0]["batch"]), tensor_tree_hash(source[0]["batch"]))
        for i, entry in enumerate(entries):
            size = (i + 1) * 2
            np.testing.assert_array_equal(entry["meta"]["user_id"], np.arange(size) + 100)
            ids = list(zip(entry["meta"]["part_id"], entry["meta"]["row_offset"]))
            self.assertEqual(len(ids), len(set(ids)))
            torch.testing.assert_close(entry["batch"][1]["item_id"], torch.arange(size * 5).view(size, 5))
            nested = subset_batch(entries[-1]["batch"], np.arange(size))
            self.assertEqual(tensor_tree_hash(entry["batch"]), tensor_tree_hash(nested))
        self.assertEqual(len(meta["row_offset"]), 6)

    def test_repeated_source_rows_cannot_be_used_to_inflate_batch_size(self):
        samples = self.samples()
        samples[1]["meta"]["part_id"][:] = 0
        with self.assertRaises(ValueError):
            build_entries(samples, sizes=(2, 4, 6))

    def test_oversize_request_cannot_repeat_or_pad_examples(self):
        with self.assertRaises(ValueError):
            build_entries(self.samples(), sizes=(2, 4, 8))

    def test_mismatched_shape_or_dtype_is_rejected(self):
        samples = self.samples()
        samples[1]["batch"][1]["item_id"] = torch.zeros(2, 6, dtype=torch.long)
        with self.assertRaises(ValueError):
            build_entries(samples, sizes=(2, 4, 6))
        samples = self.samples()
        samples[1]["batch"][1]["item_id"] = samples[1]["batch"][1]["item_id"].float()
        with self.assertRaises(ValueError):
            build_entries(samples, sizes=(2, 4, 6))

    def test_content_hash_detects_tensor_change_and_bfloat16_is_supported(self):
        batch = self.samples()[0]["batch"]
        first = tensor_tree_hash(batch)
        batch[1]["item_id"][0, 0] += 1
        self.assertNotEqual(first, tensor_tree_hash(batch))
        self.assertEqual(len(tensor_tree_hash((torch.ones(2, 3, dtype=torch.bfloat16),))), 64)

    def test_metadata_dtype_drift_is_rejected_before_identifier_rounding(self):
        samples = self.samples()
        samples[1]["meta"]["user_id"] = np.array([2**60 + 1, 2**60 + 3], dtype=np.uint64)
        with self.assertRaises(ValueError):
            build_entries(samples, sizes=(2, 4, 6))


if __name__ == "__main__":
    unittest.main()
