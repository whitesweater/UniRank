"""Reusable SISA-v2 score bias for UniRank attention layers.

The module changes only pre-softmax attention scores. It does not alter the
base Q/K/V projections, residual paths, feed-forward blocks, or hard masks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn


def _reshape_heads(
    channels: torch.Tensor,
    num_heads: int,
    channel_dim: int,
) -> torch.Tensor:
    batch_size, sequence_length, _ = channels.shape
    return (
        channels.view(batch_size, sequence_length, num_heads, channel_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _rotate_pairs(channels: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    even = channels[..., 0::2]
    odd = channels[..., 1::2]
    cosine = torch.cos(phase)
    sine = torch.sin(phase)
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        dim=-1,
    ).flatten(start_dim=-2)


def _deterministic_prefix_sum(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Inclusive prefix sum without CUDA's nondeterministic float cumsum."""

    size = values.size(dim)
    result = values
    offset = 1
    while offset < size:
        zero_shape = list(result.shape)
        zero_shape[dim] = offset
        shifted = torch.cat(
            (
                result.new_zeros(zero_shape),
                result.narrow(dim, 0, size - offset),
            ),
            dim=dim,
        )
        result = result + shifted
        offset *= 2
    return result


@dataclass(frozen=True)
class SISAAttentionConfig:
    """Configuration shared by all native UniRank attention adapters."""

    enabled: bool = False
    score_dim: int = 16
    lambda_init: float = 0.1
    decay_bias: float = -5.0
    exp_clamp: float = 11.0
    decay_reference: str = "sequence_end"
    score_scale: float = 1.0
    parameter_seed: int = 20260821

    @classmethod
    def from_params(
        cls,
        params: dict,
        *,
        decay_reference: str,
    ) -> "SISAAttentionConfig":
        enabled = params.get("sisa_enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("sisa_enabled must be a YAML/Python boolean")
        return cls(
            enabled=enabled,
            score_dim=int(params.get("sisa_score_dim", 16)),
            lambda_init=float(params.get("sisa_lambda_init", 0.1)),
            decay_bias=float(params.get("sisa_decay_bias", -5.0)),
            exp_clamp=float(params.get("sisa_exp_clamp", 11.0)),
            decay_reference=decay_reference,
            score_scale=float(params.get("sisa_score_scale", 1.0)),
            parameter_seed=int(params.get("sisa_parameter_seed", 20260821)),
        )

    def for_site(self, site_index: int) -> "SISAAttentionConfig":
        return replace(self, parameter_seed=self.parameter_seed + int(site_index))

    def build(self, hidden_size: int, num_heads: int) -> "SISAScoreBias | None":
        if not self.enabled:
            return None
        return SISAScoreBias(
            hidden_size=hidden_size,
            num_heads=num_heads,
            score_dim=self.score_dim,
            lambda_init=self.lambda_init,
            decay_bias=self.decay_bias,
            exp_clamp=self.exp_clamp,
            decay_reference=self.decay_reference,
            score_scale=self.score_scale,
            parameter_seed=self.parameter_seed,
        )


class SISAScoreBias(nn.Module):
    """SISA-v2 decay-and-rotation channels expressed as an additive score bias.

    Projection weights are bare parameters and use an isolated deterministic
    generator. Enabling SISA therefore does not consume the global RNG stream
    used to initialize the unchanged backbone parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        score_dim: int = 16,
        lambda_init: float = 0.1,
        decay_bias: float = -5.0,
        exp_clamp: float = 11.0,
        decay_reference: str = "sequence_end",
        score_scale: float = 1.0,
        parameter_seed: int = 20260821,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0:
            raise ValueError("hidden_size and num_heads must be positive")
        if score_dim <= 0 or score_dim % 2:
            raise ValueError("score_dim must be a positive even integer")
        if lambda_init <= 0:
            raise ValueError("lambda_init must be positive")
        if exp_clamp <= 0:
            raise ValueError("exp_clamp must be positive")
        if decay_reference not in {"query", "sequence_end"}:
            raise ValueError("decay_reference must be 'query' or 'sequence_end'")
        if score_scale < 0:
            raise ValueError("score_scale must be non-negative")

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.score_dim = int(score_dim)
        self.decay_bias = float(decay_bias)
        self.exp_clamp = float(exp_clamp)
        self.decay_reference = str(decay_reference)
        self.score_scale = float(score_scale)
        self.parameter_seed = int(parameter_seed)

        channel_size = self.num_heads * self.score_dim
        phase_size = self.num_heads * (self.score_dim // 2)
        self.b_weight = nn.Parameter(torch.empty(channel_size, self.hidden_size))
        self.c_weight = nn.Parameter(torch.empty(channel_size, self.hidden_size))
        self.decay_weight = nn.Parameter(
            torch.empty(self.num_heads, self.hidden_size)
        )
        self.decay_projection_bias = nn.Parameter(
            torch.full((self.num_heads,), self.decay_bias)
        )
        self.phase_weight = nn.Parameter(torch.empty(phase_size, self.hidden_size))

        raw_lambda = math.log(math.expm1(float(lambda_init)))
        self.raw_head_scale = nn.Parameter(
            torch.full((self.num_heads,), raw_lambda, dtype=torch.float32)
        )
        self.init_weights()

    def _apply(self, fn):
        super()._apply(fn)
        self.raw_head_scale.data = self.raw_head_scale.data.float()
        if self.raw_head_scale.grad is not None:
            self.raw_head_scale.grad.data = self.raw_head_scale.grad.data.float()
        return self

    def init_weights(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.parameter_seed)
        for weight in (
            self.b_weight,
            self.c_weight,
            self.decay_weight,
            self.phase_weight,
        ):
            nn.init.xavier_uniform_(weight, generator=generator)
        with torch.no_grad():
            self.decay_projection_bias.fill_(self.decay_bias)

    def lambdas(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return (F.softplus(self.raw_head_scale) + 1e-8).to(
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _minimax_offset(
        cumulative_decay: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = valid_mask[:, None, :]
        has_valid = valid.any(dim=-1)
        minimum = cumulative_decay.masked_fill(~valid, torch.inf).amin(dim=-1)
        maximum = cumulative_decay.masked_fill(~valid, -torch.inf).amax(dim=-1)
        midpoint = (minimum + maximum) * 0.5
        return torch.where(has_valid, midpoint, torch.zeros_like(midpoint))

    def _final_channels(
        self,
        raw_channels: torch.Tensor,
        cumulative_decay: torch.Tensor,
        cumulative_phase: torch.Tensor,
        offset: torch.Tensor,
        *,
        inverse_decay: bool,
    ) -> torch.Tensor:
        centered = (cumulative_decay - offset.unsqueeze(-1)).clamp(
            min=-self.exp_clamp,
            max=self.exp_clamp,
        )
        if inverse_decay:
            centered = -centered
        decay = torch.exp(centered).unsqueeze(-1)
        rotated = _rotate_pairs(raw_channels.float(), cumulative_phase)
        return (decay * rotated).to(dtype=raw_channels.dtype)

    def build_augmented_channels(
        self,
        hidden_states: torch.Tensor,
        valid_mask: torch.Tensor,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, L, D]")
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError("hidden_states does not match configured hidden_size")
        if valid_mask.shape != hidden_states.shape[:2]:
            raise ValueError("valid_mask must have shape [B, L]")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")

        valid_mask = valid_mask.bool()
        dtype = hidden_states.dtype
        raw_b = _reshape_heads(
            F.linear(hidden_states, self.b_weight),
            self.num_heads,
            self.score_dim,
        )
        raw_c = _reshape_heads(
            F.linear(hidden_states, self.c_weight),
            self.num_heads,
            self.score_dim,
        )
        phase = _reshape_heads(
            F.linear(hidden_states, self.phase_weight),
            self.num_heads,
            self.score_dim // 2,
        ).float()
        log_alpha = -F.softplus(
            F.linear(
                hidden_states.float(),
                self.decay_weight.float(),
                self.decay_projection_bias.float(),
            )
        ).permute(0, 2, 1).contiguous()

        valid = valid_mask[:, None, :, None]
        raw_b = raw_b.masked_fill(~valid, 0.0)
        raw_c = raw_c.masked_fill(~valid, 0.0)
        phase = phase.masked_fill(~valid, 0.0)
        log_alpha = log_alpha.masked_fill(~valid.squeeze(-1), 0.0)

        cumulative_decay = _deterministic_prefix_sum(log_alpha, dim=2)
        cumulative_phase = _deterministic_prefix_sum(phase, dim=2)
        offset = self._minimax_offset(cumulative_decay, valid_mask)
        if self.decay_reference == "query":
            query_decay = cumulative_decay
        else:
            valid_heads = valid_mask[:, None, :]
            has_valid = valid_heads.any(dim=-1)
            sequence_end = cumulative_decay.masked_fill(
                ~valid_heads,
                torch.inf,
            ).amin(dim=-1)
            sequence_end = torch.where(
                has_valid,
                sequence_end,
                torch.zeros_like(sequence_end),
            )
            query_decay = sequence_end.unsqueeze(-1).expand_as(cumulative_decay)

        query_channels = self._final_channels(
            raw_c,
            query_decay,
            cumulative_phase,
            offset,
            inverse_decay=False,
        )
        key_channels = self._final_channels(
            raw_b,
            cumulative_decay,
            cumulative_phase,
            offset,
            inverse_decay=True,
        )
        scale = (float(head_dim) ** 0.25) * torch.sqrt(
            self.lambdas(dtype=dtype, device=hidden_states.device)
        )
        scale = scale.view(1, -1, 1, 1)
        return query_channels * scale, key_channels * scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        valid_mask: torch.Tensor,
        head_dim: int,
        *,
        query_slice: slice | None = None,
        key_slice: slice | None = None,
    ) -> torch.Tensor:
        query_channels, key_channels = self.build_augmented_channels(
            hidden_states,
            valid_mask,
            head_dim,
        )
        if query_slice is not None:
            query_channels = query_channels[:, :, query_slice, :]
        if key_slice is not None:
            key_channels = key_channels[:, :, key_slice, :]
        score_bias = torch.matmul(
            query_channels,
            key_channels.transpose(-1, -2),
        ) / math.sqrt(float(head_dim))
        return score_bias * self.score_scale
