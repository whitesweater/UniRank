from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch

from model_zoo.HyFormer import (
    HyFormerBlock,
    QueryDecodingLayer,
    SequenceRepresentationLayer,
)
from model_zoo.UltraHSTU import (
    SequentialTransductionUnit,
    UnifiedInteractionBlocks,
)
from model_zoo.UniMixer import UniMixingLiteBlocks, UniMixingLiteLayer
from unirank.pytorch.layers import SISAAttentionConfig, SISAScoreBias


def _common_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if "sisa_score_bias" not in name
    }


def _assert_common_initialization(
    testcase: unittest.TestCase,
    baseline: torch.nn.Module,
    variant: torch.nn.Module,
) -> None:
    baseline_state = _common_state(baseline)
    variant_state = _common_state(variant)
    testcase.assertEqual(baseline_state.keys(), variant_state.keys())
    for name, expected in baseline_state.items():
        torch.testing.assert_close(variant_state[name], expected)


def _dense_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    score_mod=None,
    block_mask=None,
    scale=None,
) -> torch.Tensor:
    del block_mask
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    if score_mod is not None:
        batches, heads, query_length, key_length = scores.shape
        rows = []
        for batch_index in range(batches):
            head_rows = []
            for head_index in range(heads):
                query_rows = []
                for query_index in range(query_length):
                    query_rows.append(
                        torch.stack(
                            [
                                score_mod(
                                    scores[
                                        batch_index,
                                        head_index,
                                        query_index,
                                        key_index,
                                    ],
                                    batch_index,
                                    head_index,
                                    query_index,
                                    key_index,
                                )
                                for key_index in range(key_length)
                            ]
                        )
                    )
                head_rows.append(torch.stack(query_rows))
            rows.append(torch.stack(head_rows))
        scores = torch.stack(rows)
    return torch.matmul(torch.softmax(scores, dim=-1), value)


class ExtendedSISAAdapterTest(unittest.TestCase):
    def setUp(self):
        self.zero_config = SISAAttentionConfig(
            enabled=True,
            score_dim=4,
            score_scale=0.0,
        )
        self.enabled_config = SISAAttentionConfig(
            enabled=True,
            score_dim=4,
            score_scale=1.0,
        )

    def test_disabled_adapters_register_no_sisa_parameters(self):
        modules = [
            SequenceRepresentationLayer(input_dim=8, num_heads=2),
            QueryDecodingLayer(input_dim=8, num_heads=2),
            UniMixingLiteLayer(token_dim=8, num_token=4),
            SequentialTransductionUnit(token_dim=8, num_heads=2),
        ]
        for module in modules:
            with self.subTest(module=type(module).__name__):
                self.assertIsNone(module.sisa_score_bias)
                self.assertFalse(
                    any("sisa" in name for name, _ in module.named_parameters())
                )

    def test_zero_scale_controls_are_exact(self):
        sequence = torch.randn(2, 5, 8)
        sequence_mask = torch.tensor(
            [[False, True, True, True, True], [True, True, True, False, False]]
        )
        global_tokens = torch.randn(2, 2, 8)
        mixer_x = torch.randn(2, 4, 8)
        mixer_y = torch.randn(2, 4, 8)

        cases = []
        torch.manual_seed(101)
        sequence_baseline = SequenceRepresentationLayer(8, 2)
        torch.manual_seed(101)
        sequence_control = SequenceRepresentationLayer(
            8,
            2,
            sisa_config=self.zero_config,
        )
        cases.append(
            (
                sequence_baseline,
                sequence_control,
                lambda module: module(sequence, sequence_mask),
            )
        )

        torch.manual_seed(102)
        query_baseline = QueryDecodingLayer(8, 2)
        torch.manual_seed(102)
        query_control = QueryDecodingLayer(8, 2, sisa_config=self.zero_config)
        cases.append(
            (
                query_baseline,
                query_control,
                lambda module: module(global_tokens, sequence, sequence_mask),
            )
        )

        torch.manual_seed(103)
        mixer_baseline = UniMixingLiteLayer(token_dim=8, num_token=4)
        torch.manual_seed(103)
        mixer_control = UniMixingLiteLayer(
            token_dim=8,
            num_token=4,
            sisa_config=self.zero_config,
        )
        cases.append(
            (
                mixer_baseline,
                mixer_control,
                lambda module: module(mixer_x, mixer_y),
            )
        )

        for baseline, control, forward in cases:
            with self.subTest(module=type(baseline).__name__):
                baseline.eval()
                control.eval()
                _assert_common_initialization(self, baseline, control)
                baseline_output = forward(baseline)
                control_output = forward(control)
                baseline_tensors = (
                    baseline_output
                    if isinstance(baseline_output, tuple)
                    else (baseline_output,)
                )
                control_tensors = (
                    control_output
                    if isinstance(control_output, tuple)
                    else (control_output,)
                )
                for expected, actual in zip(baseline_tensors, control_tensors):
                    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

        torch.manual_seed(104)
        hstu_baseline = SequentialTransductionUnit(token_dim=8, num_heads=2)
        torch.manual_seed(104)
        hstu_control = SequentialTransductionUnit(
            token_dim=8,
            num_heads=2,
            sisa_config=self.zero_config,
        )
        _assert_common_initialization(self, hstu_baseline, hstu_control)
        hstu_input = torch.randn(2, 6, 8)
        hstu_mask = torch.ones(2, 6, dtype=torch.bool)
        hstu_baseline._get_sparse_block_mask = lambda *args, **kwargs: None
        hstu_control._get_sparse_block_mask = lambda *args, **kwargs: None
        with patch(
            "model_zoo.UltraHSTU.flex_attention",
            side_effect=_dense_flex_attention,
        ):
            expected = hstu_baseline(hstu_input, hstu_mask)
            actual = hstu_control(hstu_input, hstu_mask)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_enabled_adapters_have_finite_nonzero_sisa_gradients(self):
        sequence = torch.randn(2, 5, 8)
        sequence_mask = torch.tensor(
            [[False, True, True, True, True], [True, True, True, False, False]]
        )
        global_tokens = torch.randn(2, 2, 8)
        mixer_x = torch.randn(2, 4, 8)
        mixer_y = torch.randn(2, 4, 8)

        modules_and_forward = [
            (
                SequenceRepresentationLayer(
                    8,
                    2,
                    sisa_config=self.enabled_config,
                ),
                lambda module: module(sequence, sequence_mask),
            ),
            (
                QueryDecodingLayer(8, 2, sisa_config=self.enabled_config),
                lambda module: module(global_tokens, sequence, sequence_mask),
            ),
            (
                UniMixingLiteLayer(
                    token_dim=8,
                    num_token=4,
                    sisa_config=self.enabled_config,
                ),
                lambda module: module(mixer_x, mixer_y),
            ),
        ]
        for module, forward in modules_and_forward:
            with self.subTest(module=type(module).__name__):
                output = forward(module)
                tensors = output if isinstance(output, tuple) else (output,)
                sum(tensor.square().mean() for tensor in tensors).backward()
                self._assert_sisa_gradients(module)

        hstu = SequentialTransductionUnit(
            token_dim=8,
            num_heads=2,
            sisa_config=self.enabled_config,
        )
        hstu._get_sparse_block_mask = lambda *args, **kwargs: None
        hstu_input = torch.randn(2, 6, 8)
        hstu_mask = torch.ones(2, 6, dtype=torch.bool)
        with patch(
            "model_zoo.UltraHSTU.flex_attention",
            side_effect=_dense_flex_attention,
        ):
            hstu(hstu_input, hstu_mask).square().mean().backward()
        self._assert_sisa_gradients(hstu)

    def test_backbones_create_all_expected_sisa_sites(self):
        hyformer = HyFormerBlock(
            input_dim=8,
            num_heads=2,
            num_layers=2,
            num_ns_token=2,
            num_global_token=1,
            sequence_sisa_config=self.enabled_config,
            query_sisa_config=self.enabled_config,
        )
        unimixer = UniMixingLiteBlocks(
            token_dim=8,
            num_token=4,
            num_layers=2,
            sisa_config=self.enabled_config,
        )
        ultrahstu = UnifiedInteractionBlocks(
            token_dim=8,
            num_layers=2,
            truncation_start_layer=1,
            truncation_ratio=0.5,
            k1=[2, 2],
            k2=[1, 1],
            num_heads=2,
            sisa_config=self.enabled_config,
        )
        expected_counts = ((hyformer, 4), (unimixer, 2), (ultrahstu, 2))
        for module, expected_count in expected_counts:
            with self.subTest(module=type(module).__name__):
                count = sum(
                    isinstance(child, SISAScoreBias) for child in module.modules()
                )
                self.assertEqual(count, expected_count)

    def test_ultrahstu_augmented_qk_matches_additive_sisa_scores(self):
        config = SISAAttentionConfig(
            enabled=True,
            score_dim=4,
            score_scale=0.6,
        )
        module = SequentialTransductionUnit(
            token_dim=8,
            num_heads=2,
            sisa_config=config,
        )
        hidden = torch.randn(2, 6, 8)
        valid = torch.tensor(
            [[True, True, True, True, True, True],
             [True, True, True, True, False, True]]
        )
        normalized = module.pre_norm(hidden)
        _, query, key, _ = module.pre_proj(normalized).chunk(4, dim=-1)
        query = query.view(2, 6, 2, 4).transpose(1, 2)
        key = key.view(2, 6, 2, 4).transpose(1, 2)
        query_channels, key_channels = (
            module.sisa_score_bias.build_augmented_channels(
                normalized,
                valid,
                module.head_dim,
            )
        )
        channel_scale = math.sqrt(config.score_scale)
        augmented_query = torch.cat(
            (query, query_channels * channel_scale),
            dim=-1,
        )
        augmented_key = torch.cat(
            (key, key_channels * channel_scale),
            dim=-1,
        )
        actual = torch.matmul(
            augmented_query,
            augmented_key.transpose(-1, -2),
        ) / math.sqrt(module.head_dim)
        expected = (
            torch.matmul(query, key.transpose(-1, -2))
            / math.sqrt(module.head_dim)
            + module.sisa_score_bias(normalized, valid, module.head_dim)
        )
        torch.testing.assert_close(actual, expected)

    def _assert_sisa_gradients(self, module: torch.nn.Module) -> None:
        parameters = [
            parameter
            for name, parameter in module.named_parameters()
            if "sisa_score_bias" in name
        ]
        self.assertTrue(parameters)
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in parameters
            )
        )
        total_gradient = sum(parameter.grad.abs().sum() for parameter in parameters)
        self.assertGreater(total_gradient.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
