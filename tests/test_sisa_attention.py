import unittest

import torch

from unirank.pytorch.layers import MultiHeadTargetAttention, SISAAttentionConfig, SISAScoreBias, ScaledDotProductAttention


def _common_state(module):
    return {name: value for name, value in module.state_dict().items()
            if "sisa_score_bias" not in name}


class SISAScoreBiasTest(unittest.TestCase):
    def test_masked_values_do_not_enter_cumulative_am_pm(self):
        module = SISAScoreBias(8, 2, score_dim=4)
        hidden = torch.randn(2, 5, 8)
        mask = torch.tensor([[False, True, True, True, True], [True, False, True, False, True]])
        changed = hidden.clone()
        changed[~mask] = 10000.0
        torch.testing.assert_close(module(changed, mask, 4), module(hidden, mask, 4), rtol=0, atol=0)

    def test_bias_is_finite_and_backpropagates(self):
        lens = SISAScoreBias(
            hidden_size=8,
            num_heads=2,
            score_dim=4,
            decay_reference="sequence_end",
        )
        hidden = torch.randn(3, 5, 8, requires_grad=True)
        valid = torch.tensor(
            [
                [False, True, True, True, True],
                [True, True, True, False, False],
                [True, True, True, True, True],
            ]
        )
        bias = lens(hidden, valid, head_dim=4)
        self.assertEqual(bias.shape, (3, 2, 5, 5))
        self.assertTrue(torch.isfinite(bias).all())
        bias.square().mean().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in lens.parameters()
            )
        )

    def test_per_head_bias_weight_is_learnable(self):
        lens = SISAScoreBias(
            hidden_size=8,
            num_heads=2,
            score_dim=4,
            lambda_init=0.1,
        )
        optimizer = torch.optim.SGD([lens.raw_head_scale], lr=0.5)
        hidden = torch.randn(2, 5, 8)
        valid = torch.ones(2, 5, dtype=torch.bool)
        before = lens.lambdas(dtype=torch.float32, device=hidden.device).detach().clone()
        loss = lens(hidden, valid, head_dim=4).square().mean()
        loss.backward()
        self.assertIsNotNone(lens.raw_head_scale.grad)
        self.assertTrue(torch.isfinite(lens.raw_head_scale.grad).all())
        self.assertGreater(lens.raw_head_scale.grad.abs().sum().item(), 0.0)
        optimizer.step()
        after = lens.lambdas(dtype=torch.float32, device=hidden.device).detach()
        self.assertFalse(torch.equal(before, after))

    def test_hard_mask_cannot_be_reopened_by_positive_bias(self):
        attention = ScaledDotProductAttention()
        query = torch.zeros(1, 1, 1, 2)
        key = torch.zeros(1, 1, 2, 2)
        value = torch.tensor([[[[3.0, 5.0], [100.0, 200.0]]]])
        valid = torch.tensor([[[[True, False]]]])
        score_bias = torch.tensor([[[[0.0, 1e4]]]])
        output, _ = attention(
            query,
            key,
            value,
            mask=valid,
            score_bias=score_bias,
        )
        torch.testing.assert_close(output, value[..., :1, :])

    def test_target_attention_zero_scale_is_exact_off_control(self):
        torch.manual_seed(17)
        baseline = MultiHeadTargetAttention(
            input_dim=8,
            attention_dim=8,
            num_heads=2,
        )
        torch.manual_seed(17)
        zero_control = MultiHeadTargetAttention(
            input_dim=8,
            attention_dim=8,
            num_heads=2,
            sisa_config=SISAAttentionConfig(
                enabled=True,
                score_dim=4,
                score_scale=0.0,
            ),
        )
        baseline.eval()
        zero_control.eval()
        self.assertEqual(_common_state(baseline).keys(), _common_state(zero_control).keys())
        for name, expected in _common_state(baseline).items():
            torch.testing.assert_close(_common_state(zero_control)[name], expected)

        target = torch.randn(2, 8)
        history = torch.randn(2, 5, 8)
        mask = torch.tensor(
            [[False, True, True, True, True], [True, True, True, False, False]]
        )
        torch.testing.assert_close(
            baseline(target, history, mask),
            zero_control(target, history, mask),
        )
