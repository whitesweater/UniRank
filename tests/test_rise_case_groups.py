"""CPU regression tests for semantic-group mass and position interventions."""

from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import probe_rise_case_groups as probe


def softmax(value):
    value = np.asarray(value, dtype=np.float64)
    weight = np.exp(value - value.max())
    return weight / weight.sum()


class RiseCaseGroupsTest(unittest.TestCase):
    def test_mass_shift_preserves_conditionals_and_hits_mass(self):
        logits = np.array([.2, -.6, .9, 1.2, -.1])
        original = softmax(logits)
        group = np.array([0, 2])
        adjusted = logits.copy()
        adjusted[group] += probe.mass_shift(original[group].sum(), .2)
        weights = softmax(adjusted)
        self.assertAlmostEqual(weights[group].sum(), .2, places=14)
        np.testing.assert_allclose(weights[group] / weights[group].sum(), original[group] / original[group].sum())
        outside = np.array([1, 3, 4])
        np.testing.assert_allclose(weights[outside] / weights[outside].sum(), original[outside] / original[outside].sum())

    def test_equal_mass_uses_same_feasible_dose_and_conserves_probability(self):
        before = np.array([.05, .05, .35, .55])
        after = np.array([.15, .25, .04, .56])
        shifts, receipt = probe.group_interventions(before, after, [0, 1], [2], np.ones(4, dtype=bool))
        dose = receipt["equal_mass_requested_dose"]
        self.assertAlmostEqual(dose, .036)
        related = softmax(np.log(after) + shifts["related_equal_mass"])
        control = softmax(np.log(after) + shifts["control_equal_mass"])
        self.assertAlmostEqual(after[:2].sum() - related[:2].sum(), dose, places=14)
        self.assertAlmostEqual(after[2] - control[2], dose, places=14)
        self.assertAlmostEqual(related[2], after[2], places=14)
        self.assertAlmostEqual(control[:2].sum(), after[:2].sum(), places=14)
        self.assertAlmostEqual(related[3], control[3], places=14)
        self.assertAlmostEqual(related[3] - after[3], dose, places=14)
        self.assertAlmostEqual(related.sum(), 1.)
        self.assertAlmostEqual(control.sum(), 1.)
        reverted = softmax(np.log(after) + shifts["related_revert"])
        self.assertAlmostEqual(reverted[:2].sum(), before[:2].sum(), places=14)

    def test_shifts_do_not_depend_on_row_constant_score_gauge(self):
        scores = np.array([.1, .2, .7, .4])
        delta = np.array([.9, .8, -.1, -.2])
        p0 = softmax(scores)
        p1 = softmax(scores + delta)
        baseline, _ = probe.group_interventions(p0, p1, [0, 1], [2], np.ones(4, bool))
        shifted, _ = probe.group_interventions(softmax(scores + 123), softmax(scores + delta - 71),
                                                [0, 1], [2], np.ones(4, bool))
        for route in baseline:
            np.testing.assert_allclose(shifted[route], baseline[route], atol=1e-13, rtol=0)

    def test_invalid_and_empty_groups_and_infeasible_mass_rejected(self):
        allowed = np.array([1, 1, 0], dtype=bool)
        for group in ([], [0, 0], [2], [0, 1], [-1], [3], [True], [1.0]):
            with self.subTest(group=group), self.assertRaises(ValueError):
                probe.checked_positions(group, allowed)
        for current, target in ((0, .5), (1, .5), (.5, 0), (.5, 1), (np.nan, .3)):
            with self.subTest(current=current, target=target), self.assertRaises(ValueError):
                probe.mass_shift(current, target)
        for p0, p1, group, control in (([.4, .3, .3], [.4, .3, .3], [0], [1]),
                                      ([.2, .4, .4], [.5, .2, .3], [0], [0]),
                                      ([.2, .4, .4], [.5, 0., .5], [0], [1])):
            with self.assertRaises(ValueError):
                probe.group_interventions(p0, p1, group, control, np.ones(3, bool))
        with self.assertRaises(ValueError):
            probe.group_interventions([.2, .8], [.5, .5], [0], [1], np.ones(2, bool))

    def test_position_template_excludes_self_and_other_masks_and_matches_rms(self):
        deltas = np.array([[5., -3., 1., 100.], [1., 4., 2., 200.], [3., 2., 4., 300.],
                           [900., -800., 700., 600.]])
        masks = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=bool)
        result, info = probe.position_template(deltas, masks, 0)
        self.assertEqual(info["peer_indices"], [1, 2])
        self.assertEqual(result[3], deltas[0, 3])
        self.assertAlmostEqual(result[:3].mean(), deltas[0, :3].mean())
        self.assertAlmostEqual(result[:3].std(), deltas[0, :3].std())
        expected_direction = deltas[1:3, :3].mean(0)
        np.testing.assert_allclose((result[:3] - result[:3].mean()) / result[:3].std(),
                                   (expected_direction - expected_direction.mean()) / expected_direction.std())
        changed = deltas.copy()
        changed[0] *= 7
        _, changed_info = probe.position_template(changed, masks, 0)
        self.assertEqual(info["template_centered_rms_before_rescaling"], changed_info["template_centered_rms_before_rescaling"])
        with self.assertRaises(ValueError):
            probe.position_template(deltas[:1], masks[:1], 0)

    def test_bf16_rounding_is_visible_in_captured_actual_mass(self):
        q = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
        k = torch.zeros(1, 1, 3, 1, dtype=torch.bfloat16)
        bias = torch.tensor([[[[1., 0., -1.]]]], dtype=torch.bfloat16)
        shift = probe.mass_shift(float(softmax([1., 0., -1.])[0]), .31)
        adjusted = (bias.float() + torch.tensor([[[[shift, 0., 0.]]]])).to(torch.bfloat16)
        raw = probe.semantic.reconstruct_attention(q, k, adjusted, torch.ones_like(bias), 1.)
        self.assertNotEqual(float(raw["weight_after"][0, 0, 0]), .31)
        np.testing.assert_allclose(raw["weight_after"][0, 0], softmax(adjusted.float().numpy().reshape(-1)), rtol=1e-6)

    def test_real_compiled_hook_keeps_qkv_and_nonprobe_rows_unchanged(self):
        class Target(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dot_attention = probe.semantic.ScaledDotProductAttention()

            def forward(self, q, k, v, mask, bias):
                return self.dot_attention(q, k, v, scale=1., mask=mask, score_bias=bias)[0]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention_layers = torch.compile(Target(), backend="inductor")
                self.other_site = Target()
                self.eval()

            def _get_amp_context(self):
                return nullcontext()

            def forward(self, batch):
                value = (self.attention_layers(*batch) + self.other_site(*batch)).reshape(-1) / 2
                return {f"{label}_pred": value for label in probe.semantic.LABELS}

        q = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1)
        k = torch.tensor([1., 0.]).view(1, 1, 2, 1).expand(4, -1, -1, -1)
        mask = torch.ones(4, 1, 1, 2)
        bias = torch.tensor([0., 2.]).view(1, 1, 1, 2).expand(4, -1, -1, -1)
        samples = [{"batch": (q, k, k.clone(), mask, bias), "selected_indices": np.array([0, 1, 2, 3])}]
        zero = np.zeros((4, 1, 1, 2), dtype=np.float32)
        adjustment = zero.copy()
        adjustment[2, 0, 0, 1] = -1.3
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        with tempfile.TemporaryDirectory() as temporary, patch.object(probe.semantic, "load_model", side_effect=lambda *_: Model()):
            plain, _, plain_info = probe.run_route({"seed": 3}, Path("unused"), samples, "plain", None, Path(temporary))
            full, _, full_info = probe.run_route({"seed": 3}, Path("unused"), samples, "full_hook", [zero], Path(temporary))
            changed, captured, changed_info = probe.run_route({"seed": 3}, Path("unused"), samples, "related_revert", [adjustment], Path(temporary))
        self.assertEqual(plain_info["full_prediction_sha256"], full_info["full_prediction_sha256"])
        self.assertEqual(full_info["qkv_signatures"], changed_info["qkv_signatures"])
        for label in probe.semantic.LABELS:
            np.testing.assert_array_equal(plain[0][label][[0, 1, 3]], changed[0][label][[0, 1, 3]])
            self.assertNotEqual(full[0][label][2], changed[0][label][2])
        expected = probe.semantic.reconstruct_attention(q, k, bias + torch.from_numpy(adjustment), mask, 1.)
        for key in expected:
            np.testing.assert_array_equal(captured[0][key], expected[key])


if __name__ == "__main__":
    unittest.main()
