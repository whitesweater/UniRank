"""CPU checks for sample identity, historical batching and resumable exports."""

import importlib.util
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_predictions.py"
SPEC = importlib.util.spec_from_file_location("rise_export", SCRIPT)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class PredictionExportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("data", "user_info", "item_info"):
            (self.root / name).mkdir()
        (self.root / "meta_data.json").write_text(json.dumps({"action_vocab": {"exposure": 1, "Like": 2}}))
        for part, count in [(0, 5), (1, 3)]:
            frame = pd.DataFrame({"user_index": [1] * count, "item_index": [3] * count,
                                  "seq_len": np.arange(count), "user_id": [101 + part] * count})
            for label in EXPORT.LABELS:
                frame[label] = np.arange(count) % 2
            frame.to_parquet(self.root / "data" / f"part-{part:05d}.parquet")
            pd.DataFrame({"user_index": [1], "full_item_seq": [[1, 0, 2, 3, 1]],
                          "full_action_seq": [[2, 0, 1, 2, 2]]}).to_parquet(
                              self.root / "user_info" / f"part-{part:05d}.parquet")
            pd.DataFrame({"item_index": [1, 2, 3], "item_id": [901, 902, 903]}).to_parquet(
                self.root / "item_info" / f"part-{part:05d}.parquet")
        self.params = {"data_root": str(self.root), "dataset_id": EXPORT.DATASET,
                       "embedding_dim": 4, "group_id": "user_id", "feature_cols": [
                           {"name": name, "type": "meta"} for name in ("user_index", "item_index", "seq_len")
                       ] + [{"name": "user_id", "type": "categorical", "vocab_size": 1000},
                            {"name": "item_id", "type": "categorical", "source": "item", "vocab_size": 1000},
                            {"name": "action", "type": "categorical", "source": "action", "vocab_size": 4}],
                       "label_col": [{"name": label} for label in EXPORT.LABELS]}

    def dataset(self, cls=None, rank=0, world_size=1):
        return (cls or EXPORT.IdentifiedDataset)(
            str(self.root / "data"), str(self.root / "user_info"), str(self.root / "item_info"),
            batch_size=2, shuffle=False, distributed=world_size > 1, rank=rank,
            world_size=world_size, drop_last=True,
        )

    def collator(self, dataset):
        feature_map = EXPORT.feature_map_from_config(self.params)
        return EXPORT.IdentifiedCollator(EXPORT.BlockedBatchCollator(
            feature_map, 3, dataset.column_index, self.root / "user_info", self.root / "item_info"))

    def metadata(self, start=0, part=0, labels=(0, 1)):
        size = len(labels)
        meta = {column: np.arange(size, dtype=np.int64) for column in EXPORT.META_COLUMNS}
        meta["row_offset"] = np.arange(start, start + size, dtype=np.int64)
        meta["part_id"] = np.full(size, part, dtype=np.int32)
        for label in EXPORT.LABELS:
            meta[label] = np.asarray(labels, dtype=np.uint8)
        return meta

    def predictions(self, values=(0.2, 0.8)):
        return {label: np.asarray(values, dtype=np.float32) for label in EXPORT.LABELS}

    def test_original_batching_and_two_rank_partition_are_preserved(self):
        identities = set()
        for rank in (0, 1):
            original = list(self.dataset(EXPORT.BlockedParquetBatchDataset, rank, 2))
            identified = list(self.dataset(rank=rank, world_size=2))
            self.assertEqual(len(original), len(identified))
            for old, new in zip(original, identified):
                np.testing.assert_array_equal(old["rows"], new["rows"])
                self.assertEqual(old["part_id"], new["part_id"])
                for offset in range(new["row_offset"], new["row_offset"] + len(new["rows"])):
                    key = (new["part_id"], offset)
                    self.assertNotIn(key, identities)
                    identities.add(key)
        self.assertEqual(identities, {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)})

    def test_worker_partition_does_not_renumber_source_rows(self):
        dataset = self.dataset()
        with patch.object(EXPORT, "get_worker_info", return_value=type("Worker", (), {"id": 1, "num_workers": 2})()):
            batches = list(dataset)
        self.assertEqual([(batch["part_id"], batch["row_offset"]) for batch in batches], [(1, 0)])

    def test_original_collator_and_target_identity_are_preserved(self):
        dataset = self.dataset()
        collator = self.collator(dataset)
        payload = list(dataset)[1]
        batch, meta = collator(payload)
        expected = collator.original(payload)
        for old, new in zip(expected[:2], batch[:2]):
            for key in old:
                torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
        torch.testing.assert_close(expected[2], batch[2], rtol=0, atol=0)
        np.testing.assert_array_equal(meta["row_offset"], [2, 3])
        np.testing.assert_array_equal(meta["user_id"], [101, 101])
        np.testing.assert_array_equal(meta["item_id"], [903, 903])
        np.testing.assert_array_equal(meta["seq_len"], [2, 3])
        # The internal zero is not counted as valid history.
        np.testing.assert_array_equal(meta["effective_length"], [1, 2])

    def test_resume_requires_identical_manifest(self):
        directory = self.root / "export"
        EXPORT.prepare_manifest(directory, {"checkpoint_sha256": "abc", "limit": 2}, False)
        with self.assertRaises(FileExistsError):
            EXPORT.prepare_manifest(directory, {"checkpoint_sha256": "abc", "limit": 2}, False)
        EXPORT.prepare_manifest(directory, {"checkpoint_sha256": "abc", "limit": 2}, True)
        with self.assertRaises(ValueError):
            EXPORT.prepare_manifest(directory, {"checkpoint_sha256": "changed", "limit": 2}, True)

    def test_shard_receipt_detects_duplicates_label_drift_and_corruption(self):
        meta = self.metadata()
        receipt = EXPORT.write_shard(self.root, meta, self.predictions())
        path = self.root / receipt["file"]
        EXPORT.verify_shard(path, EXPORT.identity_hash(meta))
        with self.assertRaises(FileExistsError):
            EXPORT.write_shard(self.root, meta, self.predictions())
        changed = dict(meta, Like=np.array([1, 0], dtype=np.uint8))
        with self.assertRaises(ValueError):
            EXPORT.verify_shard(path, EXPORT.identity_hash(changed))
        with path.open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaises(ValueError):
            EXPORT.verify_shard(path)

    def test_global_auc_is_not_average_of_rank_auc(self):
        # Each shard ranks its own positive higher, but a cross-shard pair fails.
        a = EXPORT.write_shard(self.root, self.metadata(part=0), self.predictions((0.1, 0.2)))
        b = EXPORT.write_shard(self.root, self.metadata(part=1), self.predictions((0.8, 0.9)))
        values = EXPORT.global_metrics([self.root / a["file"], self.root / b["file"]])
        self.assertAlmostEqual(values["Like"]["AUC"], 0.75)
        self.assertEqual(values["Like"]["rows"], 4)
        self.assertEqual(values["Like"]["positive"], 2)

    def test_coverage_rejects_overlap_gaps_and_missing_parts(self):
        receipts = [{"part_id": 0, "row_start": 0, "rows": 2},
                    {"part_id": 0, "row_start": 2, "rows": 2},
                    {"part_id": 1, "row_start": 0, "rows": 2}]
        self.assertEqual(EXPORT.validate_receipts(receipts, {0: 4, 1: 2}, 2, False), 6)
        for bad in (receipts + receipts[:1], receipts[1:], receipts[:2]):
            with self.assertRaises(ValueError):
                EXPORT.validate_receipts(bad, {0: 4, 1: 2}, 2, False)

    def test_invalid_probabilities_do_not_commit_shards(self):
        with self.assertRaises(ValueError):
            EXPORT.write_shard(self.root, self.metadata(), self.predictions((float("nan"), 0.5)))
        self.assertEqual(list(self.root.glob("*.parquet")), [])

    def test_model_restore_requires_all_checkpoint_keys(self):
        class TinyHyFormer(torch.nn.Linear):
            def __init__(self, feature_map, **kwargs):
                super().__init__(1, 1)
                self._torch_compile_enabled = True

        module = importlib.import_module("model_zoo.HyFormer")
        checkpoint = self.root / "tiny.model"
        torch.save({"weight": torch.zeros(1, 1)}, checkpoint)
        with patch.object(module, "HyFormer", TinyHyFormer):
            with self.assertRaises(RuntimeError):
                EXPORT.load_model(self.params, checkpoint, -1)
            torch.save({"weight": torch.ones(1, 1), "bias": torch.zeros(1)}, checkpoint)
            model = EXPORT.load_model(self.params, checkpoint, -1)
            self.assertFalse(model.training)
            torch.testing.assert_close(model.weight, torch.ones(1, 1))

    def test_extra_baseline_requires_matching_non_sisa_protocol(self):
        reference = {"model": "HyFormer", "dataset_id": EXPORT.DATASET, "seed": 20262028,
                     "batch_size": 16384, "sisa_enabled": True, "sisa_score_dim": 16, "model_id": "E",
                     "model_root": "./checkpoints/"}
        baseline = {"model": "HyFormer", "dataset_id": EXPORT.DATASET, "seed": 20262028,
                    "batch_size": 16384, "model_id": "A",
                    "model_root": "./artifacts/rise_sample_analysis/training_checkpoints/"}
        EXPORT.validate_parameters(baseline, reference, "A")
        with self.assertRaises(ValueError):
            EXPORT.validate_parameters(dict(baseline, seed=20262027), reference, "A")
        with self.assertRaises(ValueError):
            EXPORT.validate_parameters(dict(baseline, sisa_enabled=True), reference, "A")

    def test_extra_baseline_path_has_two_explicit_allowed_directories(self):
        with patch.object(EXPORT, "ROOT", self.root):
            for directory in ("checkpoints", "artifacts/rise_sample_analysis/training_checkpoints"):
                path = self.root / directory / EXPORT.DATASET / "native.log"
                self.assertEqual(EXPORT.validate_extra_log(path), path.resolve())
            for path in (self.root / "unrelated" / EXPORT.DATASET / "native.log",
                         self.root / "checkpoints/QK_Video_Action/native.log",
                         self.root / "checkpoints/MerRec_Action/../elsewhere/native.log"):
                with self.assertRaises(ValueError):
                    EXPORT.validate_extra_log(path)


if __name__ == "__main__":
    unittest.main()
