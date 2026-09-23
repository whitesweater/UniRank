import unittest
from unittest.mock import patch

import torch
from torch import nn

from model_zoo.HyFormer import HyFormerBlock
from unirank.pytorch.layers import SISAAttentionConfig
from unirank.pytorch.models.rank_model import BaseModel

from scripts.run_sisa_memory_budget import install_layerwise_compilation, parse_arguments


class SISAMemoryBudgetTest(unittest.TestCase):
    def test_real_compile_wrappers_preserve_seeded_reset_weights(self):
        def model():
            torch.manual_seed(17)
            instance = BaseModel.__new__(BaseModel)
            nn.Module.__init__(instance)
            instance._torch_compile_enabled = False
            instance.enable_torch_compile = True
            instance.before = nn.Linear(8, 8)
            instance.unified_interaction_layers = HyFormerBlock(
                input_dim=8, num_heads=2, num_layers=3, num_ns_token=4, num_global_token=2,
                sequence_sisa_config=SISAAttentionConfig(enabled=True, score_dim=64),
                query_sisa_config=SISAAttentionConfig(enabled=True, score_dim=64))
            instance.after = nn.Linear(8, 8)
            return instance

        baseline = model()
        baseline._maybe_enable_torch_compile()
        torch.manual_seed(91)
        baseline.reset_parameters()
        expected_rng = torch.random.get_rng_state().clone()
        candidate = model()
        original = install_layerwise_compilation()
        self.addCleanup(setattr, BaseModel, "_maybe_enable_torch_compile", original)
        candidate._maybe_enable_torch_compile()
        torch.manual_seed(91)
        candidate.reset_parameters()
        self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
        expected = {name.replace("_orig_mod.", ""): value for name, value in baseline.state_dict().items()}
        actual = {name.replace("_orig_mod.", ""): value for name, value in candidate.state_dict().items()}
        self.assertEqual(actual.keys(), expected.keys())
        for name in expected:
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_layerwise_compilation_preserves_parameter_identity_and_initialization_order(self):
        model = BaseModel.__new__(BaseModel)
        nn.Module.__init__(model)
        model._torch_compile_enabled = False
        model.enable_torch_compile = True
        model.before = nn.Linear(8, 8)
        model.unified_interaction_layers = HyFormerBlock(
            input_dim=8, num_heads=2, num_layers=3, num_ns_token=4, num_global_token=2)
        model.after = nn.Linear(8, 8)
        modules = list(model._modules)
        parameters = [id(value) for value in model.parameters()]
        original = install_layerwise_compilation()
        self.addCleanup(setattr, BaseModel, "_maybe_enable_torch_compile", original)
        with patch("torch.compile", side_effect=lambda module, **kwargs: module) as compile_call:
            model._maybe_enable_torch_compile()
        self.assertEqual(list(model._modules), modules)
        self.assertEqual([id(value) for value in model.parameters()], parameters)
        self.assertEqual(compile_call.call_count, 11)
        self.assertTrue(model._torch_compile_enabled)

    def test_formal_run_cannot_be_silently_truncated_by_a_probe(self):
        arguments = ["--activation-memory-budget", "0", "--probe-steps", "3", "--",
                     "--run-id", "SISA_score_dim_HyFormer_MerRec_Action_r64_seed20262028_attempt3"]
        with self.assertRaises(ValueError):
            parse_arguments(arguments)
        arguments[-1] = "SISA_score_dim_memory_probe_budget0"
        options, forwarded = parse_arguments(arguments)
        self.assertEqual(options.probe_steps, 3)
        self.assertEqual(forwarded, arguments[-2:])

    def test_formal_wrapper_preserves_training_arguments(self):
        args = ["--config", "./config", "--sisa-score-dim", "64", "--batch-size", "16384"]
        options, forwarded = parse_arguments(["--activation-memory-budget", "0", "--", *args])
        self.assertEqual(options.probe_steps, 0)
        self.assertEqual(options.activation_memory_budget, 0)
        self.assertEqual(forwarded, args)
        for invalid in ("-1", "1.1", "nan"):
            with self.assertRaises(ValueError):
                parse_arguments(["--activation-memory-budget", invalid, "--", *args])


if __name__ == "__main__":
    unittest.main()
