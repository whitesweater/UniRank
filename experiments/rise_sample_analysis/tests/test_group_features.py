"""CPU-only tests for sample identity, causal windows, and target frequencies."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'experiments/rise_sample_analysis/scripts/build_group_features.py'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GroupFeaturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module(SCRIPT, 'group_features')
        cls.preprocess = load_module(
            ROOT / 'data/MerRec/preprocess_MerRec_seq_action.py', 'merrec_preprocess')

    def test_window_excludes_target_future_padding_and_unknown_categories(self):
        result = self.mod.compute_history_features(
            np.array([[0, 1, 2, 3], [0, 0, 0, 0], [0, 0, 1, 4]]),
            np.array([[0, 1, 2, 1], [0, 0, 0, 0], [0, 0, 1, 2]]),
            np.array([0, 1, 1, 0, 3]), np.array([1, 1, 3]),
            np.array([[0, 0], [1, 0], [0, 1]]), ['Like', 'Cart'],
            4, {0, 3}, {3},
        )
        np.testing.assert_array_equal(result['effective_length'], [3, 0, 2])
        np.testing.assert_array_equal(result['c1_known_count'], [2, 0, 1])
        np.testing.assert_array_equal(result['c1_ambiguous_count'], [0, 0, 1])
        np.testing.assert_allclose(result['c1_entropy'][[0, 2]], [0, 0])
        self.assertTrue(np.isnan(result['c1_entropy'][1]))
        np.testing.assert_allclose(result['target_c1_history_fraction'][0], 1)
        self.assertTrue(np.isnan(result['target_c1_history_fraction'][1:]).all())
        np.testing.assert_array_equal(result['history_Like_count'], [2, 0, 1])
        np.testing.assert_array_equal(result['history_Cart_count'], [1, 0, 1])

    def test_vectorized_window_uses_last_history_and_preserves_target_anchor(self):
        seq = {'offsets': np.array([0, 5]), 'items': np.array([1, 2, 3, 4, 5]),
               'actions': np.array([1, 2, 1, 2, 1]), 'user_lookup': np.array([0])}
        items, actions = self.mod.gather_windows(seq, np.array([0, 0]),
                                                np.array([0, 4]), np.array([1, 5]), 2)
        np.testing.assert_array_equal(items, [[0, 0], [3, 4]])
        np.testing.assert_array_equal(actions, [[0, 0], [1, 2]])
        with self.assertRaisesRegex(ValueError, 'target'):
            self.mod.gather_windows(seq, np.array([0]), np.array([4]), np.array([4]), 2)

    def test_counts_global_ids_only_from_train_targets_and_row_ids_are_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds, out = base / 'dataset', base / 'output'
            self._fixture(ds)
            self.mod.main(['--dataset-dir', str(ds), '--output-dir', str(out),
                           'count-train', '--batch-size', '2'])
            users = np.load(out / 'frequencies/user_train_target_count.npy')
            items = np.load(out / 'frequencies/item_train_target_count.npy')
            np.testing.assert_array_equal(users, [0, 2, 1])
            np.testing.assert_array_equal(items, [0, 1, 1, 1, 0])
            selected = base / 'selected.parquet'
            pq.write_table(pa.table({'part_id': [0, 0], 'row_offset': [2, 0]}), selected)
            self.mod.main(['--dataset-dir', str(ds), '--output-dir', str(out),
                           'build', '--row-ids', str(selected), '--retained-batch-size', '0',
                           '--batch-size', '2'])
            actual = pq.read_table(out / 'features/part-00000.parquet').to_pydict()
            self.assertEqual(actual['row_offset'], [0, 2])
            self.assertEqual(actual['item_id'], [2, 4])
            self.assertEqual(actual['item_train_target_count'], [1, 0])
            self.assertEqual(actual['effective_length'], [1, 0])
            self.assertEqual(actual['history_Like_count'], [1, 0])
            self.assertEqual(actual['history_action_temporal_status'],
                             ['unverified_eventual_outcomes', 'no_history'])
            self.assertIsNone(actual['c1_entropy'][1])
            self.assertNotIn(3, actual['row_offset'])
            manifest = json.loads((out / 'frequencies/manifest.json').read_text())
            self.assertEqual(manifest['train_rows_counted'], 3)
            self.assertTrue(manifest['complete'])

    def test_retained_coverage_rejects_selected_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds, out = base / 'dataset', base / 'output'
            self._fixture(ds)
            self.mod.main(['--dataset-dir', str(ds), '--output-dir', str(out), 'count-train'])
            selected = base / 'selected.parquet'
            pq.write_table(pa.table({'part_id': [0], 'row_offset': [3]}), selected)
            with self.assertRaisesRegex(ValueError, 'outside retained coverage'):
                self.mod.main(['--dataset-dir', str(ds), '--output-dir', str(out), 'build',
                               '--row-ids', str(selected), '--retained-batch-size', '3'])

    def test_partial_frequency_arrays_are_not_formal_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds, out = base / 'dataset', base / 'output'
            self._fixture(ds)
            self.mod.main(['--dataset-dir', str(ds), '--output-dir', str(out),
                           'count-train', '--limit-rows', '1'])
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                self.mod.main(['--dataset-dir', str(ds), '--output-dir', str(out), 'build'])

    def test_duplicate_row_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ids.parquet'
            pq.write_table(pa.table({'part_id': [0, 0], 'row_offset': [1, 1]}), path)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                self.mod.read_row_ids(path)

    def test_conservative_category_policy_marks_legacy_oov_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp)
            (ds / 'feature_vocab.json').write_text(json.dumps(
                {'c1_id': {'0': 0, '1': 1, '__OOV__': 2}}, indent=4))
            policy = self.mod.category_policy(ds, {'vocab_size': {'c1_id': 3}})
            self.assertEqual(policy['unknown_ids'], [0, 2])
            self.assertEqual(policy['ambiguous_ids'], [2])

    def test_eventual_outcome_is_in_history_before_the_action_occurs(self):
        p = self.preprocess
        rows = []
        for item, time, action, row_id in [('a', 100, 'exposure', 0),
                                         ('b', 200, 'exposure', 1), ('a', 300, 'Like', 2)]:
            rows.append({'user_id_raw': 'u', 'item_id_raw': item, 'stime_ms': time,
                         '_row_id': row_id, 'action': p.RAW_ACTION_VOCAB[action],
                         **{label: float(label == action) for label in p.LABEL_COLUMNS}})
        patterns, vocab = p.build_action_maps()
        anchors = p.build_view_anchor_samples(pd.DataFrame(rows), patterns, vocab)
        prior = anchors[anchors['stime_ms'] < 200].iloc[0]
        self.assertEqual(prior['Like'], 1)
        self.assertEqual(prior['action'], vocab['Like'])
        # This proves a possible information dependency, not its observed prevalence.
        self.assertLess(prior['stime_ms'], 200)

    def test_source_row_order_is_not_equivalent_to_event_timestamp_order(self):
        p = self.preprocess
        rows = []
        for time, action, row_id in [(100, 'exposure', 10), (50, 'Like', 11)]:
            rows.append({'user_id_raw': 'u', 'item_id_raw': 'a', 'stime_ms': time,
                         '_row_id': row_id, 'action': p.RAW_ACTION_VOCAB[action],
                         **{label: float(label == action) for label in p.LABEL_COLUMNS}})
        patterns, vocab = p.build_action_maps()
        anchors = p.build_view_anchor_samples(pd.DataFrame(rows).sort_values('stime_ms'),
                                             patterns, vocab)
        self.assertEqual(anchors.iloc[0]['Like'], 1)

    def _fixture(self, ds):
        ds.mkdir()
        meta = {'vocab_size': {'user_id': 3, 'item_id': 5, 'c1_id': 3},
                'label': ['Like', 'Cart'], 'action_vocab': {'Like': 1, 'exposure': 2},
                'oov_filter': {'rule': 'feature values with count < min_feat_count are mapped to 0'}}
        (ds / 'meta_data.json').write_text(json.dumps(meta))
        for split, part in [('train', 0), ('train', 1), ('test', 0)]:
            for kind in ['data', 'item_info', 'user_info']:
                (ds / split / kind).mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({'item_index': [0, 1, 2, 3, 4],
                                     'item_id': [0, 1, 2, 3, 4] if part == 0 else [0, 3, 1, 2, 4],
                                     'c1_id': [0, 1, 2, 1, 0]}),
                           ds / split / 'item_info' / f'part-{part:05d}.parquet')
        pq.write_table(pa.table({'user_id': [1, 1], 'item_index': [1, 2]}),
                       ds / 'train/data/part-00000.parquet')
        pq.write_table(pa.table({'user_id': [2], 'item_index': [1]}),
                       ds / 'train/data/part-00001.parquet')
        pq.write_table(pa.table({'user_index': [0, 0, 1, 1], 'user_id': [1, 1, 2, 2],
                                 'item_index': [2, 3, 4, 1], 'seq_len': [1, 2, 0, 1]}),
                       ds / 'test/data/part-00000.parquet')
        pq.write_table(pa.table({'user_index': [0, 1], 'full_item_seq': [[1, 2, 3], [4, 1]],
                                 'full_action_seq': [[1, 2, 1], [2, 1]],
                                 'full_timestamp_seq': [[100, 200, 300], [100, 200]]}),
                       ds / 'test/user_info/part-00000.parquet')


if __name__ == '__main__':
    unittest.main()
