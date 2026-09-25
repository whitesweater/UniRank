"""Process-local targeted RISE controls; importing does not alter core code.

fixed_cumulative changes only input-conditioned control increments to learned
input-independent head constants. Query/sequence-end references, prefix scan,
minimax centering, endpoint clipping, content projections and head scales remain.
target_only/field_only retain native full RISE only at the selected HiFormer
reference sites; removed score-bias modules are genuinely None.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
import torch.nn.functional as F

from unirank.pytorch.layers.attentions.sisa import (
    SISAAttentionConfig, SISAScoreBias, _deterministic_prefix_sum, _reshape_heads,
)

MODES = ("fixed_cumulative", "target_only", "field_only")


class FixedCumulativeScoreBias(SISAScoreBias):
    """Learned constant increments at both native reference types, not equal capacity."""

    def __init__(self, **kwargs):
        if kwargs.get("variant", "full") != "full":
            raise ValueError("Fixed controls require the full component configuration")
        super().__init__(**kwargs)
        self.control = "fixed_cumulative"
        self.decay_weight.requires_grad_(False)
        self.phase_weight.requires_grad_(False)
        # Same zero phase increment and retained decay bias as earlier RankMixer F.
        self.fixed_phase_increment = nn.Parameter(torch.zeros(self.num_heads, self.score_dim // 2))

    def control_increments(self, hidden_states, valid_mask):
        batch, length, _ = hidden_states.shape
        log_alpha = -F.softplus(self.decay_projection_bias.float())
        log_alpha = log_alpha[None, :, None].expand(batch, -1, length)
        phase = self.fixed_phase_increment.float()[None, :, None, :].expand(batch, -1, length, -1)
        valid = valid_mask.bool()[:, None, :]
        return log_alpha.masked_fill(~valid, 0.), phase.masked_fill(~valid.unsqueeze(-1), 0.)

    def build_augmented_channels(self, hidden_states, valid_mask, head_dim):
        if hidden_states.ndim != 3 or hidden_states.size(-1) != self.hidden_size:
            raise ValueError("hidden_states must have shape [B,L,hidden_size]")
        if valid_mask.shape != hidden_states.shape[:2] or head_dim <= 0:
            raise ValueError("Invalid mask or head dimension")
        valid_mask = valid_mask.bool()
        raw_b = _reshape_heads(F.linear(hidden_states, self.b_weight), self.num_heads, self.score_dim)
        raw_c = _reshape_heads(F.linear(hidden_states, self.c_weight), self.num_heads, self.score_dim)
        valid = valid_mask[:, None, :, None]
        raw_b = raw_b.masked_fill(~valid, 0.)
        raw_c = raw_c.masked_fill(~valid, 0.)
        log_alpha, phase = self.control_increments(hidden_states, valid_mask)
        cumulative_decay = _deterministic_prefix_sum(log_alpha, dim=2)
        cumulative_phase = _deterministic_prefix_sum(phase, dim=2)
        offset = self._minimax_offset(cumulative_decay, valid_mask)
        if self.decay_reference == "query":
            query_decay = cumulative_decay
        else:
            valid_heads = valid_mask[:, None, :]
            has_valid = valid_heads.any(dim=-1)
            endpoint = cumulative_decay.masked_fill(~valid_heads, torch.inf).amin(dim=-1)
            endpoint = torch.where(has_valid, endpoint, torch.zeros_like(endpoint))
            query_decay = endpoint.unsqueeze(-1).expand_as(cumulative_decay)
        query = self._final_channels(raw_c, query_decay, cumulative_phase, offset, inverse_decay=False)
        key = self._final_channels(raw_b, cumulative_decay, cumulative_phase, offset, inverse_decay=True)
        scale = (float(head_dim) ** .25) * torch.sqrt(
            self.lambdas(dtype=hidden_states.dtype, device=hidden_states.device))
        return query * scale[None, :, None, None], key * scale[None, :, None, None]


@contextmanager
def patched_build(mode):
    if mode not in MODES:
        raise ValueError(f"Unsupported targeted control: {mode}")
    original = SISAAttentionConfig.build

    def build(config, hidden_size, num_heads):
        if not config.enabled:
            return original(config, hidden_size, num_heads)
        if config.variant != "full":
            raise ValueError("Targeted controls cannot be combined with B/C/D component modes")
        if config.decay_reference not in {"query", "sequence_end"}:
            raise ValueError("Unknown reference type in targeted adapter construction")
        if mode == "target_only":
            return original(config, hidden_size, num_heads) if config.decay_reference == "query" else None
        if mode == "field_only":
            return original(config, hidden_size, num_heads) if config.decay_reference == "sequence_end" else None
        return FixedCumulativeScoreBias(
            hidden_size=hidden_size, num_heads=num_heads, score_dim=config.score_dim,
            lambda_init=config.lambda_init, decay_bias=config.decay_bias, exp_clamp=config.exp_clamp,
            decay_reference=config.decay_reference, score_scale=config.score_scale,
            parameter_seed=config.parameter_seed, variant=config.variant)

    SISAAttentionConfig.build = build
    try:
        yield
    finally:
        SISAAttentionConfig.build = original


def unwrapped(module):
    while hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


def parameter_counts(model):
    active = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    return {"stored_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "active_parameters": sum(parameter.numel() for _, parameter in active),
            "active_dense_parameters": sum(parameter.numel() for name, parameter in active if "embedding" not in name),
            "active_embedding_parameters": sum(parameter.numel() for name, parameter in active if "embedding" in name)}


def audit_model(model, mode, expected_dense=None):
    """Read-only post-construction check; does not draw RNG or alter parameters."""
    if mode not in MODES:
        raise ValueError("Unknown targeted audit mode")
    if mode == "fixed_cumulative":
        block = unwrapped(model.unified_interaction_layers)
        slots = [(f"seq_layers.{index}", unwrapped(layer).sisa_score_bias, "sequence_end", True)
                 for index, layer in enumerate(block.seq_layers)]
        slots += [(f"decode_layers.{index}", unwrapped(layer).sisa_score_bias, "query", True)
                  for index, layer in enumerate(block.decode_layers)]
        if len(slots) != 6:
            raise ValueError("Historical HyFormer fixed study requires six attention sites")
    else:
        block = unwrapped(model.unified_interaction_layers)
        slots = [("attention_layers", unwrapped(model.attention_layers).sisa_score_bias, "query", mode == "target_only")]
        slots += [(f"mixer_layers.{index}", unwrapped(layer).sisa_score_bias, "sequence_end", mode == "field_only")
                  for index, layer in enumerate(block.mixer_layers)]
        if len(slots) != 4:
            raise ValueError("Historical HiFormer study requires one target and three field sites")
    rows = []
    for name, module, reference, enabled in slots:
        if (module is not None) != enabled:
            raise ValueError(f"Unexpected enabled/disabled site: {name}")
        row = {"site": name, "reference": reference, "enabled": enabled}
        if enabled:
            expected_type = FixedCumulativeScoreBias if mode == "fixed_cumulative" else SISAScoreBias
            if type(module) is not expected_type or module.decay_reference != reference:
                raise ValueError(f"Unexpected adapter semantics at {name}")
            row.update(class_name=type(module).__name__, parameter_seed=module.parameter_seed,
                       hidden_size=module.hidden_size, heads=module.num_heads, score_dim=module.score_dim,
                       stored=sum(parameter.numel() for parameter in module.parameters()),
                       active=sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))
            if mode == "fixed_cumulative" and (module.decay_weight.requires_grad or module.phase_weight.requires_grad):
                raise ValueError("Input-dependent control weights must be frozen")
        rows.append(row)
    counts = parameter_counts(model)
    if expected_dense is not None and counts["active_dense_parameters"] != expected_dense:
        raise ValueError(f"Active dense count mismatch: {counts['active_dense_parameters']} != {expected_dense}")
    return {"mode": mode, "sites": rows, "enabled_site_count": sum(row["enabled"] for row in rows), **counts}
