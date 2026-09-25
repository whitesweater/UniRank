from __future__ import annotations

from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import torch

from controls import ControlScoreBias, parameter_counts, patched_build
from run_rankmixer_controls import validate_protocol
from unirank.pytorch.layers.attentions.sisa import SISAAttentionConfig, SISAScoreBias
from unirank.pytorch.layers.attentions.target_attention import MultiHeadTargetAttention


class RISEControlsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(381)
        self.kwargs = dict(hidden_size=8, num_heads=2, score_dim=4,
                           decay_reference="query", parameter_seed=281)
        self.hidden = torch.randn(2, 5, 8)
        self.valid = torch.ones(2, 5, dtype=torch.bool)

    def test_full_patch_preserves_class_state_output_and_rng(self):
        config = SISAAttentionConfig(enabled=True, decay_reference="query", score_dim=4)
        before = torch.random.get_rng_state().clone()
        expected = config.build(8, 2)
        with patched_build("full"):
            actual = config.build(8, 2)
        self.assertIs(type(actual), SISAScoreBias)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(set(expected.state_dict()), set(actual.state_dict()))
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(value, actual.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(expected(self.hidden, self.valid, 4),
                                   actual(self.hidden, self.valid, 4), rtol=0, atol=0)

    def test_local_is_exact_active_parameter_match_and_shared_initialization(self):
        full = SISAScoreBias(**self.kwargs)
        local = ControlScoreBias(control="local", **self.kwargs)
        self.assertEqual(parameter_counts(full), parameter_counts(local))
        self.assertTrue(all(p.requires_grad for p in local.parameters()))
        for name, value in full.state_dict().items():
            torch.testing.assert_close(value, local.state_dict()[name], rtol=0, atol=0)

    def test_local_pair_has_no_nonlocal_input_or_padding_dependence(self):
        local = ControlScoreBias(control="local", **self.kwargs)
        original = local(self.hidden, self.valid, 4)[:, :, 4, 0]
        altered = self.hidden.clone()
        altered[:, 1:4] = 100 * torch.randn_like(altered[:, 1:4])
        valid = self.valid.clone()
        valid[:, 2] = False
        torch.testing.assert_close(original, local(altered, valid, 4)[:, :, 4, 0],
                                   rtol=0, atol=0)
        hidden = self.hidden.clone().requires_grad_(True)
        local(hidden, self.valid, 4)[:, :, 4, 0].sum().backward()
        self.assertEqual(hidden.grad[:, 1:4].abs().max().item(), 0.0)
        full = SISAScoreBias(**self.kwargs)
        self.assertGreater((full(self.hidden, self.valid, 4)[:, :, 4, 0] -
                            full(altered, valid, 4)[:, :, 4, 0]).abs().max().item(), 1e-5)

    def test_local_all_parameter_tensors_have_finite_nonzero_gradients(self):
        local = ControlScoreBias(control="local", **self.kwargs)
        output = local(self.hidden, self.valid, 4)
        (output * torch.randn_like(output)).sum().backward()
        for name, parameter in local.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_fixed_increments_are_input_independent_with_expected_active_count(self):
        fixed = ControlScoreBias(control="fixed", **self.kwargs)
        first = fixed.control_increments(self.hidden, self.valid)
        second = fixed.control_increments(20 * torch.randn_like(self.hidden), self.valid)
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        d, h, r = 8, 2, 4
        self.assertEqual(parameter_counts(fixed)["active"], 2 * h * r * d + h * (r // 2 + 2))
        self.assertFalse(fixed.decay_weight.requires_grad)
        self.assertFalse(fixed.phase_weight.requires_grad)
        output = fixed(self.hidden, self.valid, 4)
        (output * torch.randn_like(output)).sum().backward()
        for name, parameter in fixed.named_parameters():
            if parameter.requires_grad:
                with self.subTest(parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0.0)
            else:
                self.assertIsNone(parameter.grad)

    def test_zero_padding_and_query_key_slices(self):
        valid = self.valid.clone()
        valid[:, 1] = False
        for control in ("local", "fixed"):
            module = ControlScoreBias(control=control, **self.kwargs)
            full = module(self.hidden, valid, 4)
            sliced = module(self.hidden, valid, 4, query_slice=slice(-1, None),
                            key_slice=slice(0, -1))
            torch.testing.assert_close(sliced, full[:, :, -1:, :-1])
            self.assertEqual(full[:, :, 1, :].abs().max().item(), 0.0)
            self.assertEqual(full[:, :, :, 1].abs().max().item(), 0.0)

    def test_patch_is_temporary_and_rejects_sequence_end_and_component_modes(self):
        original = SISAAttentionConfig.build
        for control in ("local", "fixed"):
            with patched_build(control):
                self.assertIsNone(SISAAttentionConfig(enabled=False).build(8, 2))
                with self.assertRaises(ValueError):
                    SISAAttentionConfig(enabled=True).build(8, 2)
                with self.assertRaises(ValueError):
                    SISAAttentionConfig(enabled=True, decay_reference="query",
                                        variant="content_only").build(8, 2)
            self.assertIs(SISAAttentionConfig.build, original)

    def test_actual_rankmixer_site_counts_and_native_initialization_are_matched(self):
        config = SISAAttentionConfig(enabled=True, decay_reference="query", score_dim=16,
                                     parameter_seed=20260822)
        torch.manual_seed(20262028)
        full = MultiHeadTargetAttention(input_dim=96, attention_dim=256, sisa_config=config)
        self.assertEqual(parameter_counts(full.sisa_score_bias)["active"], 3938)
        for control, active in (("local", 3938), ("fixed", 3082)):
            torch.manual_seed(20262028)
            with patched_build(control):
                adapter = MultiHeadTargetAttention(input_dim=96, attention_dim=256,
                                                    sisa_config=config)
            self.assertEqual(parameter_counts(adapter.sisa_score_bias)["active"], active)
            for name, value in full.state_dict().items():
                if "sisa_score_bias" not in name:
                    torch.testing.assert_close(value, adapter.state_dict()[name], rtol=0, atol=0)
            output = adapter(torch.randn(3, 96), torch.randn(3, 6, 96),
                             torch.ones(3, 6, dtype=torch.bool))
            self.assertEqual(output.shape, (3, 96))
            self.assertTrue(torch.isfinite(output).all())

    def test_both_controls_match_all_frozen_e2028_configuration_fields(self):
        for control in ("local", "fixed"):
            params = validate_protocol(control, 1)
            self.assertEqual(params["sisa_score_dim"], 16)
            self.assertEqual(params["seed"], 20262028)
            self.assertEqual(params["batch_size"] * params["world_size"], 32768)


if __name__ == "__main__":
    unittest.main()
