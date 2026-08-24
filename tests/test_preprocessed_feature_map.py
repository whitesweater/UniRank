from __future__ import annotations

import unittest

from unirank.features import feature_map_from_config
from unirank.utils import load_config


class PreprocessedFeatureMapTest(unittest.TestCase):
    def test_builds_taobao_schema_without_vocab_materialization(self):
        params = load_config("./config", "RankMixer_Taobao_Action")
        feature_map = feature_map_from_config(params)
        self.assertEqual(feature_map.dataset_id, "Taobao_Action")
        self.assertEqual(feature_map.labels, ["is_click", "cart", "fav", "buy"])
        self.assertEqual(feature_map.features["item_id"]["vocab_size"], 846812)
        self.assertEqual(feature_map.features["item_id"]["source"], "item")
        self.assertEqual(feature_map.features["action"]["source"], "action")
        self.assertGreater(feature_map.input_length, feature_map.num_fields)

    def test_requires_explicit_vocab_size(self):
        params = {
            "dataset_id": "fixture",
            "data_root": ".",
            "feature_cols": [
                {"name": "item_id", "active": True, "type": "categorical"}
            ],
            "label_col": {"name": "label"},
        }
        with self.assertRaises(ValueError):
            feature_map_from_config(params)


if __name__ == "__main__":
    unittest.main()
