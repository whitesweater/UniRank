"""CPU audit of targeted control semantics, capacities and initialization."""

from copy import deepcopy
import gc
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import targeted_controls as TARGET
import run_targeted_controls as RUN
from controls import ControlScoreBias
from unirank.features import feature_map_from_config
from unirank.pytorch.layers.attentions.sisa import SISAAttentionConfig, SISAScoreBias, _deterministic_prefix_sum, _reshape_heads


class TargetedControlsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(721)
        self.hidden = torch.randn(2, 7, 8)
        self.valid = torch.tensor([[1, 1, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
        self.kwargs = dict(hidden_size=8, num_heads=2, score_dim=4, parameter_seed=823)

    def test_query_fixed_matches_existing_rankmixer_fixed_exactly(self):
        original = ControlScoreBias(control="fixed", decay_reference="query", **self.kwargs)
        targeted = TARGET.FixedCumulativeScoreBias(decay_reference="query", **self.kwargs)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, targeted.state_dict()[key], atol=0, rtol=0)
        for trained in (False, True):
            if trained:
                with torch.no_grad():
                    original.fixed_phase_increment.fill_(.12)
                    original.decay_projection_bias.copy_(torch.tensor([-.5, .7]))
                targeted.load_state_dict(original.state_dict(), strict=True)
            torch.testing.assert_close(original(self.hidden, self.valid, 4),
                                       targeted(self.hidden, self.valid, 4), atol=0, rtol=0)

    def test_sequence_end_uses_native_reference_minimax_clipping_and_phase(self):
        module = TARGET.FixedCumulativeScoreBias(decay_reference="sequence_end", exp_clamp=.15, **self.kwargs)
        with torch.no_grad():
            module.decay_projection_bias.copy_(torch.tensor([-.5, .7]))
            module.fixed_phase_increment.copy_(torch.tensor([[.1, -.2], [.2, .3]]))
        log_alpha, phase = module.control_increments(self.hidden, self.valid)
        g, phi = _deterministic_prefix_sum(log_alpha, 2), _deterministic_prefix_sum(phase, 2)
        raw_b = _reshape_heads(F.linear(self.hidden, module.b_weight), 2, 4)
        raw_c = _reshape_heads(F.linear(self.hidden, module.c_weight), 2, 4)
        valid = self.valid[:, None, :, None]
        raw_b, raw_c = raw_b.masked_fill(~valid, 0), raw_c.masked_fill(~valid, 0)
        minimum = g.masked_fill(~self.valid[:, None, :], torch.inf).amin(-1)
        maximum = g.masked_fill(~self.valid[:, None, :], -torch.inf).amax(-1)
        active = self.valid.any(-1)[:, None]
        endpoint = torch.where(active, minimum, torch.zeros_like(minimum))
        midpoint = torch.where(active, (minimum + maximum) / 2, torch.zeros_like(minimum))
        expected_q = module._final_channels(raw_c, endpoint[:, :, None].expand_as(g), phi, midpoint, inverse_decay=False)
        expected_k = module._final_channels(raw_b, g, phi, midpoint, inverse_decay=True)
        scale = (4 ** .25) * torch.sqrt(module.lambdas(dtype=self.hidden.dtype, device=self.hidden.device))
        actual_q, actual_k = module.build_augmented_channels(self.hidden, self.valid, 4)
        torch.testing.assert_close(actual_q, expected_q * scale[None, :, None, None], atol=0, rtol=0)
        torch.testing.assert_close(actual_k, expected_k * scale[None, :, None, None], atol=0, rtol=0)
        self.assertEqual(actual_q[1].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(actual_q).all())
        query_ref = TARGET.FixedCumulativeScoreBias(decay_reference="query", exp_clamp=.15, **self.kwargs)
        query_ref.load_state_dict(module.state_dict(), strict=True)
        self.assertGreater((actual_q - query_ref.build_augmented_channels(self.hidden, self.valid, 4)[0]).abs().max().item(), .001)

    def test_fixed_increments_are_input_independent_and_all_active_gradients_exist(self):
        for reference in ("query", "sequence_end"):
            module = TARGET.FixedCumulativeScoreBias(decay_reference=reference, **self.kwargs)
            first = module.control_increments(self.hidden, self.valid)
            second = module.control_increments(torch.randn_like(self.hidden) * 10, self.valid)
            for left, right in zip(first, second):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            output = module(self.hidden, self.valid, 4)
            (output * torch.randn_like(output)).sum().backward()
            for name, parameter in module.named_parameters():
                with self.subTest(reference=reference, parameter=name):
                    if name in {"decay_weight", "phase_weight"}:
                        self.assertFalse(parameter.requires_grad)
                        self.assertIsNone(parameter.grad)
                    else:
                        self.assertTrue(parameter.requires_grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(parameter.grad.abs().sum().item(), 0.)
            self.assertEqual(sum(p.numel() for p in module.parameters() if p.requires_grad), 2 * 2 * 4 * 8 + 2 * (2 + 2))

    def test_patch_restores_original_and_disabled_sites_are_none(self):
        original = SISAAttentionConfig.build
        for mode in TARGET.MODES:
            with TARGET.patched_build(mode):
                self.assertIsNone(SISAAttentionConfig(enabled=False).build(8, 2))
                for reference in ("query", "sequence_end"):
                    config = SISAAttentionConfig(enabled=True, score_dim=4, decay_reference=reference)
                    module = config.build(8, 2)
                    expected = mode == "fixed_cumulative" or (mode == "target_only" and reference == "query") or (mode == "field_only" and reference == "sequence_end")
                    self.assertEqual(module is not None, expected)
                with self.assertRaises(ValueError):
                    SISAAttentionConfig(enabled=True, variant="content_only").build(8, 2)
            self.assertIs(SISAAttentionConfig.build, original)

    def test_retained_site_branches_all_have_finite_nonzero_gradients(self):
        for mode, reference, hidden_size, heads, head_dim in (
                ("target_only", "query", 48, 1, 256),
                ("field_only", "sequence_end", 256, 2, 128)):
            config = SISAAttentionConfig(enabled=True, decay_reference=reference, score_dim=16,
                                         parameter_seed=20260822)
            with TARGET.patched_build(mode):
                module = config.build(hidden_size, heads)
            hidden = torch.randn(2, 6, hidden_size)
            mask = torch.ones(2, 6, dtype=torch.bool)
            output = module(hidden, mask, head_dim)
            (output * torch.randn_like(output)).sum().backward()
            for name, parameter in module.named_parameters():
                with self.subTest(mode=mode, parameter=name):
                    self.assertTrue(parameter.requires_grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def build_small_vocab_model(self, task_id, mode=None):
        import model_zoo
        params = deepcopy(RUN.reference_params(RUN.TASKS[task_id]))
        # Only vocabulary allocation is reduced for CPU testing. Dense structure,
        # all attention sites, dimensions and seed offsets remain the real ones.
        for column in params["feature_cols"]:
            if "vocab_size" in column:
                column["vocab_size"] = min(column["vocab_size"], 8)
        params.update(gpu=-1, distributed=False, world_size=1, rank=0, local_rank=0,
                      enable_torch_compile=False, enable_bf16=False, verbose=0)
        feature_map = feature_map_from_config(params)
        torch.manual_seed(20262028)
        if mode is None:
            model = getattr(model_zoo, params["model"])(feature_map, **params)
        else:
            with TARGET.patched_build(mode):
                model = getattr(model_zoo, params["model"])(feature_map, **params)
        return model, torch.get_rng_state().clone()

    def test_real_dense_shapes_sites_and_all_retained_initializations_match(self):
        for task_id, task in enumerate(RUN.TASKS):
            full, full_rng = self.build_small_vocab_model(task_id)
            control, control_rng = self.build_small_vocab_model(task_id, task["mode"])
            self.assertTrue(torch.equal(full_rng, control_rng))
            full_state, control_state = full.state_dict(), control.state_dict()
            native_names = {name for name in full_state if "sisa_score_bias" not in name}
            self.assertEqual(native_names, {name for name in control_state if "sisa_score_bias" not in name})
            for name, value in control_state.items():
                if name in full_state:
                    torch.testing.assert_close(value, full_state[name], atol=0, rtol=0)
                else:
                    self.assertTrue(name.endswith("fixed_phase_increment"))
                    self.assertEqual(value.abs().sum().item(), 0.)
            audit = TARGET.audit_model(control, task["mode"], task["expected_dense"])
            self.assertEqual(audit["enabled_site_count"], (6, 1, 3)[task_id])
            enabled = [site for site in audit["sites"] if site["enabled"]]
            expected_seeds = ([20260822, 20260823, 20260824, 20260922, 20260923, 20260924]
                              if task_id == 0 else ([20260822] if task_id == 1 else [20260922, 20260923, 20260924]))
            self.assertEqual([site["parameter_seed"] for site in enabled], expected_seeds)
            self.assertEqual(TARGET.parameter_counts(full)["active_dense_parameters"], RUN.expected_counts(task_id)["historical_E_active_dense"])
            del full_state, control_state, full, control
            gc.collect()

    def test_all_frozen_effective_fields_match_except_unique_model_id(self):
        for task_id, task in enumerate(RUN.TASKS):
            reference = RUN.reference_params(task)
            actual = RUN.effective_params(task_id, 1)
            self.assertEqual({key for key in reference.keys() | actual.keys() if reference.get(key) != actual.get(key)}, {"model_id"})
            counts = RUN.expected_counts(task_id)
            self.assertEqual(counts["active_dense"], task["expected_dense"])
            self.assertEqual(counts["active_total"], counts["active_dense"] + counts["active_embedding"])
            self.assertIn("--nproc_per_node=2", RUN.training_command(task_id, 1))
        self.assertEqual(RUN.expected_counts(0)["active_total"], 810076253)
        self.assertEqual(RUN.expected_counts(1)["active_total"], 167827222)
        self.assertEqual(RUN.expected_counts(2)["active_total"], 167888240)


if __name__ == "__main__":
    unittest.main()
