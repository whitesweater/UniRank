from __future__ import annotations

import unittest

import torch

from model_zoo.HiFormer import HiformerAttentionLayer
from model_zoo.OneTrans import MixedMHA, OneTransBlock
from model_zoo.Zenith import TokenwiseMultiHeadSelfAttention
from unirank.pytorch.layers import (
    MultiHeadTargetAttention,
    SISAAttentionConfig,
    SISAScoreBias,
    ScaledDotProductAttention,
)


def _common_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if "sisa_score_bias" not in name
    }


class SISAScoreBiasTest(unittest.TestCase):
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


class NativeAttentionAdapterTest(unittest.TestCase):
    def test_disabled_sisa_adds_no_parameters_at_any_native_site(self):
        modules = [
            MixedMHA(input_dim=8, num_heads=2, num_ns_token=2),
            HiformerAttentionLayer(
                token_dim=8,
                num_token=4,
                num_heads=2,
                qkv_rank=4,
            ),
            TokenwiseMultiHeadSelfAttention(
                token_dim=8,
                num_token=4,
                num_head=2,
            ),
            MultiHeadTargetAttention(
                input_dim=8,
                attention_dim=8,
                num_heads=2,
            ),
        ]
        for module in modules:
            with self.subTest(module=type(module).__name__):
                self.assertIsNone(module.sisa_score_bias)
                self.assertFalse(
                    any("sisa" in name for name, _ in module.named_parameters())
                )

    def test_hiformer_and_zenith_zero_bias_controls_are_exact(self):
        config = SISAAttentionConfig(
            enabled=True,
            score_dim=4,
            score_scale=0.0,
        )
        cases = [
            (
                HiformerAttentionLayer,
                {
                    "token_dim": 8,
                    "num_token": 4,
                    "num_heads": 2,
                    "qkv_rank": 4,
                },
            ),
            (
                TokenwiseMultiHeadSelfAttention,
                {"token_dim": 8, "num_token": 4, "num_head": 2},
            ),
        ]
        inputs = torch.randn(2, 4, 8)
        for module_class, kwargs in cases:
            with self.subTest(module=module_class.__name__):
                torch.manual_seed(31)
                baseline = module_class(**kwargs)
                torch.manual_seed(31)
                zero_control = module_class(**kwargs, sisa_config=config)
                baseline.eval()
                zero_control.eval()
                baseline_state = _common_state(baseline)
                zero_state = _common_state(zero_control)
                self.assertEqual(baseline_state.keys(), zero_state.keys())
                for name, expected in baseline_state.items():
                    torch.testing.assert_close(zero_state[name], expected)
                torch.testing.assert_close(
                    baseline(inputs),
                    zero_control(inputs),
                )

    def test_onetrans_common_initialization_and_zero_control(self):
        baseline_config = SISAAttentionConfig(enabled=False)
        zero_config = SISAAttentionConfig(
            enabled=True,
            score_dim=4,
            score_scale=0.0,
        )
        torch.manual_seed(23)
        baseline = OneTransBlock(
            input_dim=8,
            num_heads=2,
            num_layers=2,
            num_ns_token=2,
            reduction_ratio=0.5,
            sisa_config=baseline_config,
        )
        torch.manual_seed(23)
        zero_control = OneTransBlock(
            input_dim=8,
            num_heads=2,
            num_layers=2,
            num_ns_token=2,
            reduction_ratio=0.5,
            sisa_config=zero_config,
        )
        baseline.eval()
        zero_control.eval()
        baseline_state = _common_state(baseline)
        zero_state = _common_state(zero_control)
        self.assertEqual(baseline_state.keys(), zero_state.keys())
        for name, expected in baseline_state.items():
            torch.testing.assert_close(zero_state[name], expected)

        sequence = torch.randn(2, 6, 8)
        non_sequence = torch.randn(2, 2, 8)
        valid = torch.tensor(
            [[False, True, True, True, True, True], [True] * 6]
        )
        torch.testing.assert_close(
            baseline(sequence, non_sequence, valid),
            zero_control(sequence, non_sequence, valid),
        )

    def test_native_attention_sites_have_finite_sisa_gradients(self):
        query_config = SISAAttentionConfig(
            enabled=True,
            score_dim=4,
            decay_reference="query",
        )
        full_config = SISAAttentionConfig(
            enabled=True,
            score_dim=4,
            decay_reference="sequence_end",
        )
        modules_and_inputs = [
            (
                MixedMHA(
                    input_dim=8,
                    num_heads=2,
                    num_ns_token=2,
                    sisa_config=query_config,
                ),
                (
                    torch.randn(2, 5, 8),
                    torch.randn(2, 3, 8),
                    torch.randn(2, 2, 8),
                ),
                {"kv_mask": torch.ones(2, 5, dtype=torch.bool)},
            ),
            (
                HiformerAttentionLayer(
                    token_dim=8,
                    num_token=4,
                    num_heads=2,
                    qkv_rank=4,
                    sisa_config=full_config,
                ),
                (torch.randn(2, 4, 8),),
                {},
            ),
            (
                TokenwiseMultiHeadSelfAttention(
                    token_dim=8,
                    num_token=4,
                    num_head=2,
                    sisa_config=full_config,
                ),
                (torch.randn(2, 4, 8),),
                {},
            ),
        ]
        for module, args, kwargs in modules_and_inputs:
            with self.subTest(module=type(module).__name__):
                output = module(*args, **kwargs)
                tensors = output if isinstance(output, tuple) else (output,)
                loss = sum(tensor.square().mean() for tensor in tensors)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                sisa_parameters = [
                    parameter
                    for name, parameter in module.named_parameters()
                    if "sisa_score_bias" in name
                ]
                self.assertTrue(sisa_parameters)
                self.assertTrue(
                    all(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all()
                        for parameter in sisa_parameters
                    )
                )


if __name__ == "__main__":
    unittest.main()
