from collections import OrderedDict
from copy import deepcopy
import tempfile
import unittest

import torch

import model_zoo
from unirank.features import FeatureMap
from unirank.pytorch.layers import SISAScoreBias


class HistoryAdapterTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.features = FeatureMap("fixture", self.directory.name)
        self.features.default_emb_dim = 4
        self.features.labels = ["click", "conversion"]
        self.features.features = OrderedDict(
            (name, {"type": "categorical", "source": source, "vocab_size": 16,
                    "padding_idx": 0})
            for name, source in (("user_id", "user"), ("user_group", "user"),
                                 ("item_id", "item"), ("category_id", "item"),
                                 ("previous_action", "action"))
        )
        self.features.num_fields = self.features.get_num_fields()
        self.params = dict(
            gpu=-1, model_root=self.directory.name, verbose=0,
            metrics=["AUC", "logloss"], loss=["binary_crossentropy"] * 2,
            task="binary_classification", num_tasks=2, embedding_dim=4,
            token_dim=8, num_heads=2, num_layers=2, num_ns_token=4,
            num_global_token=2, qkv_rank=4, tower_hidden_units=[8],
            dense_optimizer="AdamW", sparse_optimizer="Adagrad",
            enable_torch_compile=False, enable_bf16=False,
            sisa_score_dim=4, sisa_score_scale=1.0,
        )
        self.inputs = (
            {"user_id": torch.tensor([1, 2]), "user_group": torch.tensor([2, 3])},
            {name: torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
             for name in ("item_id", "category_id", "previous_action")},
            torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]),
        )

    def build(self, name, enabled, scale=1.0):
        torch.manual_seed(57)
        params = dict(self.params, sisa_enabled=enabled, sisa_score_scale=scale)
        return getattr(model_zoo, name)(deepcopy(self.features), **params).eval()

    def predict(self, model):
        outputs = model(self.inputs)
        return torch.cat([outputs[label + "_pred"] for label in self.features.labels], dim=-1)

    def test_only_paper_history_sites_are_enabled(self):
        expected = {
            "HiFormer": ["attention_layers.sisa_score_bias"],
            "RankMixer": ["attention_layers.sisa_score_bias"],
            "Zenith": ["attention_layers.sisa_score_bias"],
            "OneTrans": [f"unified_interaction_layers.mha_layers.{i}.sisa_score_bias"
                         for i in range(2)],
            "HyFormer": [f"unified_interaction_layers.{stage}.{i}.sisa_score_bias"
                         for stage in ("seq_layers", "decode_layers") for i in range(2)],
        }
        for name, sites in expected.items():
            with self.subTest(model=name):
                model = self.build(name, True)
                actual = [path for path, module in model.named_modules()
                          if isinstance(module, SISAScoreBias)]
                self.assertEqual(actual, sites)

    def test_disabled_adds_no_parameters_and_enabled_preserves_backbone_rng(self):
        for name in ("HiFormer", "HyFormer", "OneTrans", "RankMixer", "Zenith"):
            with self.subTest(model=name):
                baseline = self.build(name, False)
                baseline_rng = torch.random.get_rng_state().clone()
                enabled = self.build(name, True)
                self.assertTrue(torch.equal(baseline_rng, torch.random.get_rng_state()))
                self.assertFalse(any("sisa" in key for key in baseline.state_dict()))
                common = {key: value for key, value in enabled.state_dict().items()
                          if "sisa_score_bias" not in key}
                self.assertEqual(common.keys(), baseline.state_dict().keys())
                for key, value in baseline.state_dict().items():
                    torch.testing.assert_close(common[key], value, rtol=0, atol=0)

    def test_zero_scale_restores_native_forward_exactly(self):
        for name in ("HiFormer", "HyFormer", "OneTrans", "RankMixer", "Zenith"):
            with self.subTest(model=name):
                expected = self.predict(self.build(name, False))
                actual = self.predict(self.build(name, True, 0.0))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_all_enabled_sites_receive_finite_gradients(self):
        for name in ("HiFormer", "HyFormer", "OneTrans", "RankMixer", "Zenith"):
            with self.subTest(model=name):
                model = self.build(name, True)
                output = self.predict(model)
                self.assertTrue(torch.isfinite(output).all())
                output.square().sum().backward()
                parameters = [parameter for key, parameter in model.named_parameters()
                              if "sisa_score_bias" in key and parameter.requires_grad]
                self.assertTrue(parameters)
                for parameter in parameters:
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
