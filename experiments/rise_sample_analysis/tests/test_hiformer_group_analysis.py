"""Small-table checks for matched HiFormer grouping and clustered uncertainty."""

import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import analyze_hiformer_groups as GROUPS


class HiFormerGroupAnalysisTest(unittest.TestCase):
    def test_macro_keeps_shared_label_user_covariance(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        users = np.repeat(np.arange(4), 2)
        a = np.array([.1, .4, .6, .8, .3, .2, .7, .5])
        e = np.array([.1, .6, .3, .8, .2, .4, .7, .5])
        forward = GROUPS.summarize_label(y, a, e, users)
        duplicate = GROUPS.summarize_macro([forward, forward])
        self.assertGreater(forward[0]["auc_cluster_se"], 0)
        self.assertAlmostEqual(duplicate["auc_cluster_se"], forward[0]["auc_cluster_se"])
        reverse = GROUPS.summarize_label(y, e, a, users)
        cancellation = GROUPS.summarize_macro([forward, reverse])
        self.assertAlmostEqual(cancellation["delta_auc"], 0)
        self.assertAlmostEqual(cancellation["auc_cluster_se"], 0)
        self.assertAlmostEqual(cancellation["logloss_cluster_se"], 0)

    def test_single_class_label_is_not_removed_from_macro_denominator(self):
        users = np.array([1, 1, 2, 2])
        good = GROUPS.summarize_label(np.array([0, 1, 0, 1]), np.array([.1, .4, .5, .3]),
                                       np.array([.1, .5, .2, .8]), users)
        absent = GROUPS.summarize_label(np.ones(4), np.full(4, .3), np.full(4, .4), users)
        self.assertIsNone(absent[0]["baseline_auc"])
        self.assertEqual(absent[0]["positives"], 4)
        self.assertIsNone(GROUPS.summarize_macro([good, absent])["delta_auc"])

    def test_single_user_preserves_effect_point_without_interval(self):
        row, _ = GROUPS.summarize_label(np.array([0, 1]), np.array([.8, .2]),
                                        np.array([.2, .8]), np.array([1, 1]))
        self.assertEqual(row["delta_auc"], 1)
        self.assertIsNone(row["auc_cluster_se"])

    def test_frequency_entity_quantiles_and_zero_ties(self):
        counts = np.array([0, 0, 1, 1, 1, 10, 100], dtype=np.uint64)
        cuts = GROUPS.frequency_cuts(counts)
        self.assertEqual(cuts, (1, 10))
        groups, _ = GROUPS.frequency_groups(np.array([0, 1, 2, 10, 11]), cuts)
        np.testing.assert_array_equal(groups, [0, 1, 2, 2, 3])
        tied, _ = GROUPS.frequency_groups(np.array([0, 1, 2]), (1, 1))
        np.testing.assert_array_equal(tied, [0, 1, 3])

    def test_length_boundaries(self):
        groups, _ = GROUPS.length_groups(np.array([0, 1, 20, 21, 50, 51, 99, 100]))
        np.testing.assert_array_equal(groups, [0, 1, 1, 2, 2, 3, 3, 4])
        with self.assertRaises(ValueError):
            GROUPS.length_groups(np.array([101]))

    def test_streaming_frequency_reconstructs_block_local_item_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets/QK_Video_Action"
            (dataset / "train/data").mkdir(parents=True)
            (dataset / "train/item_info").mkdir()
            (dataset / "meta_data.json").write_text(json.dumps({"vocab_size": {"user_id": 4, "item_id": 6}}))
            for part, item_id in [(0, 3), (1, 5)]:
                filename = f"part-{part:05d}.parquet"
                pq.write_table(pa.table({"user_id": [1, 2], "item_index": [1, 1]}), dataset / "train/data" / filename)
                pq.write_table(pa.table({"item_index": [1], "item_id": [item_id]}), dataset / "train/item_info" / filename)
            args = SimpleNamespace(dataset="QK_Video_Action", output_dir=root / "output", batch_size=1)
            small = dict(GROUPS.DATASETS[args.dataset], train_rows=4)
            with patch.object(GROUPS, "ROOT", root), patch.dict(GROUPS.DATASETS, {args.dataset: small}), patch.object(
                    GROUPS, "source_hashes", return_value={"test": "frozen"}):
                GROUPS.count_train(args)
                arrays, manifest = GROUPS.validate_frequency_inputs(args.output_dir / "frequencies", args.dataset)
                np.testing.assert_array_equal(arrays["user"], [0, 2, 2, 0])
                np.testing.assert_array_equal(arrays["item"], [0, 0, 0, 2, 0, 2])
                self.assertEqual(manifest["train_rows_counted"], 4)

    def build_export(self, root, variant):
        dataset = "QK_Video_Action"
        labels = GROUPS.DATASETS[dataset]["labels"]
        directory = root / variant
        directory.mkdir(parents=True)
        n = 16384
        columns = {"part_id": np.zeros(n, np.int32), "row_offset": np.arange(n, dtype=np.int64),
                   "user_id": np.arange(n) % 4 + 1, "item_id": np.ones(n, np.int64),
                   "seq_len": np.full(n, 10, np.int64), "effective_length": np.full(n, 10, np.int16)}
        for label in labels:
            columns[label] = (np.arange(n) % 2).astype(np.uint8)
            columns[f"{label}_pred"] = np.where(columns[label], .8, .2).astype(np.float32)
        identity = GROUPS.sample_identity(columns, dataset, labels)
        shard = directory / "part-00000-offset-000000000000.parquet"
        pq.write_table(pa.table({"dataset": [dataset] * n, "split": ["test"] * n, **columns}), shard)
        receipt = {"rows": n, "part_id": 0, "row_start": 0, "sha256": GROUPS.frequencies.sha256(shard),
                   "identity_sha256": identity}
        shard.with_suffix(".json").write_text(json.dumps(receipt))
        shared = {"dataset": dataset, "model": "HiFormer", "split": "test", "labels": labels,
                  "world_size": 2, "batch_size": 16384, "expected_rows_by_part": {0: n}}
        (root / "manifest.json").write_text(json.dumps(shared))
        manifest = {"shared_manifest_sha256": GROUPS.canonical_hash(shared), "data_version": "frozen",
                    "logged_params": {"model": "HiFormer", "dataset_id": dataset, "seed": 20262028,
                                      "sisa_enabled": variant == "E", "model_id": variant}}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        complete = {"smoke_only": False, "metric_check_passed": True, "rows": n, "variant": variant,
                    "data_version": "frozen", "metrics": {label: {"AUC": 1, "logloss": .2} for label in labels},
                    "historical_deviation": {label: {"AUC": 0, "logloss": 0} for label in labels},
                    "sample_identity": {shard.name: identity}}
        (directory / "complete.json").write_text(json.dumps(complete))
        return directory

    def test_formal_pair_identity_and_tampered_labels_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.build_export(root / "before", "A")
            e = self.build_export(root / "after", "E")
            small = dict(GROUPS.DATASETS["QK_Video_Action"], test_rows=16384)
            with patch.dict(GROUPS.DATASETS, {"QK_Video_Action": small}):
                data, _ = GROUPS.read_paired_exports(a, e, "QK_Video_Action")
                self.assertEqual(len(data["row_offset"]), 16384)
                shard = next(e.glob("*.parquet"))
                table = pq.read_table(shard)
                table = table.set_column(table.schema.get_field_index("follow"), "follow", pa.array(np.ones(16384, np.uint8)))
                pq.write_table(table, shard)
                receipt = json.loads(shard.with_suffix(".json").read_text())
                receipt["sha256"] = GROUPS.frequencies.sha256(shard)
                shard.with_suffix(".json").write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "digest"):
                    GROUPS.read_paired_exports(a, e, "QK_Video_Action")

    def test_rejects_smoke_and_old_seed_even_if_success_claimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self.build_export(Path(tmp), "E")
            small = dict(GROUPS.DATASETS["QK_Video_Action"], test_rows=16384)
            with patch.dict(GROUPS.DATASETS, {"QK_Video_Action": small}):
                path = directory / "complete.json"
                complete = json.loads(path.read_text())
                complete["smoke_only"] = True
                path.write_text(json.dumps(complete))
                with self.assertRaises(ValueError):
                    GROUPS.validate_formal_export(directory, "QK_Video_Action", "E")
                complete["smoke_only"] = False
                path.write_text(json.dumps(complete))
                path = directory / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest["logged_params"]["seed"] = 20262027
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "20262028"):
                    GROUPS.validate_formal_export(directory, "QK_Video_Action", "E")

    def test_complete_small_pair_analysis_writes_frozen_results_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.build_export(root / "before", "A")
            e = self.build_export(root / "after", "E")
            dataset = root / "datasets/QK_Video_Action"
            dataset.mkdir(parents=True)
            (dataset / "meta_data.json").write_text("{}")
            freq = root / "frequencies"
            freq.mkdir()
            np.save(freq / "user_train_target_count.npy", np.array([0, 2, 2, 0, 0, 0], dtype=np.uint64))
            np.save(freq / "item_train_target_count.npy", np.array([0, 0, 0, 0, 0, 4], dtype=np.uint64))
            manifest = {"dataset_dir": str(dataset), "complete": True, "train_rows_counted": 4,
                        "metadata_sha256": GROUPS.frequencies.sha256(dataset / "meta_data.json"),
                        "sources": [], "arrays": {path.name: GROUPS.frequencies.sha256(path) for path in freq.glob("*.npy")}}
            (freq / "manifest.json").write_text(json.dumps(manifest))
            args = SimpleNamespace(dataset="QK_Video_Action", baseline_dir=a, rise_dir=e,
                                   frequency_dir=freq, output_dir=root / "analysis",
                                   dimensions=["effective_length", "user_train_target_frequency", "item_train_target_frequency"])
            small = dict(GROUPS.DATASETS[args.dataset], train_rows=4, test_rows=16384)
            with patch.object(GROUPS, "ROOT", root), patch.dict(GROUPS.DATASETS, {args.dataset: small}), patch.object(
                    GROUPS, "source_hashes", return_value={"fixture": "frozen"}):
                GROUPS.analyze(args)
            complete = json.loads((args.output_dir / "complete.json").read_text())
            self.assertTrue(complete["identity_verified"])
            self.assertEqual(complete["rows"], 16384)
            with (args.output_dir / "group_results.csv").open() as stream:
                results = list(csv.DictReader(stream))
            self.assertEqual(len(results), 20)
            self.assertTrue(all(float(row["delta_auc"]) == 0 for row in results))
            self.assertEqual(sum(row["label"] == "macro_4_tasks" for row in results), 4)


if __name__ == "__main__":
    unittest.main()
