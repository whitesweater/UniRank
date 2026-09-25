"""Study-local RISE controls; importing this file does not patch repository code.

The local control is restricted to query-reference sites. Its score for (i, j)
depends on x_i and x_j only, including numerical stabilization. The fixed
control learns input-independent per-head decay and per-pair phase increments;
it retains accumulation and is not a parameter-matched control.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch import nn

from unirank.pytorch.layers.attentions.sisa import (
    SISAAttentionConfig,
    SISAScoreBias,
    _deterministic_prefix_sum,
    _reshape_heads,
)


class ControlScoreBias(SISAScoreBias):
    """Keep native projections and all adapter APIs, alter controls only."""

    def __init__(self, *, control: str, **kwargs):
        if control not in {"local", "fixed"}:
            raise ValueError("control must be local or fixed")
        if kwargs.get("decay_reference", "sequence_end") != "query":
            raise ValueError("This study only supports query-reference sites")
        if kwargs.get("variant", "full") != "full":
            raise ValueError("Do not combine controls with component ablations")
        super().__init__(**kwargs)
        self.control = control
        if control == "fixed":
            self.decay_weight.requires_grad_(False)
            self.phase_weight.requires_grad_(False)
            self.fixed_phase_increment = nn.Parameter(
                torch.zeros(self.num_heads, self.score_dim // 2)
            )

    def control_increments(self, hidden_states, valid_mask):
        """Return masked log-decay [B,H,L] and phase [B,H,L,r/2]."""
        batch, length, _ = hidden_states.shape
        if self.control == "fixed":
            log_alpha = -F.softplus(self.decay_projection_bias.float())
            log_alpha = log_alpha[None, :, None].expand(batch, -1, length)
            phase = self.fixed_phase_increment.float()[None, :, None, :]
            phase = phase.expand(batch, -1, length, -1)
        else:
            log_alpha = -F.softplus(F.linear(
                hidden_states.float(), self.decay_weight.float(),
                self.decay_projection_bias.float(),
            )).permute(0, 2, 1).contiguous()
            phase = _reshape_heads(
                F.linear(hidden_states, self.phase_weight), self.num_heads,
                self.score_dim // 2,
            ).float()
        valid = valid_mask[:, None, :]
        return (log_alpha.masked_fill(~valid, 0.0),
                phase.masked_fill(~valid.unsqueeze(-1), 0.0))

    def build_augmented_channels(self, hidden_states, valid_mask, head_dim):
        if hidden_states.ndim != 3 or hidden_states.size(-1) != self.hidden_size:
            raise ValueError("hidden_states must have shape [B,L,hidden_size]")
        if valid_mask.shape != hidden_states.shape[:2] or head_dim <= 0:
            raise ValueError("valid_mask shape or head_dim is invalid")
        valid_mask = valid_mask.bool()
        raw_b = _reshape_heads(F.linear(hidden_states, self.b_weight),
                               self.num_heads, self.score_dim)
        raw_c = _reshape_heads(F.linear(hidden_states, self.c_weight),
                               self.num_heads, self.score_dim)
        valid = valid_mask[:, None, :, None]
        raw_b = raw_b.masked_fill(~valid, 0.0)
        raw_c = raw_c.masked_fill(~valid, 0.0)
        log_alpha, phase = self.control_increments(hidden_states, valid_mask)
        if self.control == "fixed":
            decay_state = _deterministic_prefix_sum(log_alpha, dim=2)
            phase_state = _deterministic_prefix_sum(phase, dim=2)
            offset = self._minimax_offset(decay_state, valid_mask)
        else:
            decay_state, phase_state = log_alpha, phase
            # A sequence-derived offset can make clipping nonlocal. Zero is a
            # fixed gauge; local exponents are clipped with the original bound.
            offset = decay_state.new_zeros(decay_state.shape[:2])
        query = self._final_channels(raw_c, decay_state, phase_state, offset,
                                     inverse_decay=False)
        key = self._final_channels(raw_b, decay_state, phase_state, offset,
                                   inverse_decay=True)
        scale = (float(head_dim) ** 0.25) * torch.sqrt(self.lambdas(
            dtype=hidden_states.dtype, device=hidden_states.device,
        ))
        return query * scale[None, :, None, None], key * scale[None, :, None, None]


def parameter_counts(module):
    return {"stored": sum(p.numel() for p in module.parameters()),
            "active": sum(p.numel() for p in module.parameters() if p.requires_grad)}


@contextmanager
def patched_build(control):
    """Process-local patch before construction/compilation; always restore it."""
    if control not in {"full", "local", "fixed"}:
        raise ValueError("Unknown study control")
    original = SISAAttentionConfig.build

    def build(config, hidden_size, num_heads):
        if control == "full" or not config.enabled:
            return original(config, hidden_size, num_heads)
        return ControlScoreBias(
            control=control, hidden_size=hidden_size, num_heads=num_heads,
            score_dim=config.score_dim, lambda_init=config.lambda_init,
            decay_bias=config.decay_bias, exp_clamp=config.exp_clamp,
            decay_reference=config.decay_reference, score_scale=config.score_scale,
            parameter_seed=config.parameter_seed, variant=config.variant,
        )

    SISAAttentionConfig.build = build
    try:
        yield
    finally:
        SISAAttentionConfig.build = original
