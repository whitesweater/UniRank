#!/usr/bin/env python3
"""Run the unchanged training entrypoint with a pinned compiler memory policy.

Diagnostic probes are explicitly named, stop before evaluation, and are never
accepted as experiment results. The formal path does not patch model training.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unirank.pytorch.models.rank_model import BaseModel


def parse_arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-memory-budget", type=float, required=True)
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument("--verify-equivalence", action="store_true")
    parser.add_argument("--split-hyformer-compile", action="store_true")
    options, forwarded = parser.parse_known_args(argv)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not math.isfinite(options.activation_memory_budget) or not 0 <= options.activation_memory_budget <= 1:
        raise ValueError("activation_memory_budget must be finite and in [0, 1]")
    if options.probe_steps < 0:
        raise ValueError("probe_steps cannot be negative")
    if options.probe_steps:
        run_id = (forwarded[forwarded.index("--run-id") + 1]
                  if "--run-id" in forwarded else "")
        if not run_id.startswith("SISA_score_dim_memory_probe_"):
            raise ValueError("A truncated probe requires a distinct diagnostic run ID")
    return options, forwarded


def compile_hyformer_layers(block):
    import torch
    for name in ("seq_layers", "decode_layers", "boost_layers"):
        layers = getattr(block, name)
        for index, layer in enumerate(layers):
            layers[index] = torch.compile(layer, backend="inductor")
            logging.info("Compiled HyFormer submodule: %s.%s", name, index)
    return block


def install_layerwise_compilation():
    from torch import nn
    from model_zoo.HyFormer import HyFormerBlock
    original = BaseModel._maybe_enable_torch_compile

    def compile_model(model):
        block = model._modules.get("unified_interaction_layers")
        if (model._torch_compile_enabled or not model.enable_torch_compile
                or not isinstance(block, HyFormerBlock)):
            return original(model)
        # Keep registration order and Parameter identities unchanged: optimizer
        # construction precedes this call and reset_parameters follows it.
        model._modules["unified_interaction_layers"] = nn.Identity()
        try:
            original(model)
        finally:
            model._modules["unified_interaction_layers"] = block
        compile_hyformer_layers(block)

    BaseModel._maybe_enable_torch_compile = compile_model
    return original


def verify_equivalence(budget: float, split_compile: bool = False) -> None:
    import torch
    import torch._functorch.config as compiler_config
    from model_zoo.HyFormer import HyFormerBlock
    from unirank.pytorch.layers import SISAAttentionConfig

    torch.manual_seed(20262028)
    baseline = HyFormerBlock(
        input_dim=256, num_heads=2, num_layers=3, num_ns_token=4, num_global_token=2,
        sequence_sisa_config=SISAAttentionConfig(enabled=True, score_dim=64,
            parameter_seed=20260822, decay_reference="sequence_end"),
        query_sisa_config=SISAAttentionConfig(enabled=True, score_dim=64,
            parameter_seed=20260822, decay_reference="query"))
    candidate = copy.deepcopy(baseline)
    tensors = (torch.randn(4, 10, 256, device="cuda", requires_grad=True),
               torch.randn(4, 2, 256, device="cuda", requires_grad=True),
               torch.randn(4, 4, 256, device="cuda", requires_grad=True),
               torch.tensor([[False] + [True] * 9] * 4, device="cuda"))
    cloned = tuple(value.detach().clone().requires_grad_(value.requires_grad) for value in tensors)
    with compiler_config.patch(activation_memory_budget=1.0):
        compiled = torch.compile(baseline.cuda(), backend="inductor")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = compiled(*tensors)
            loss = expected[1].float().square().mean()
        loss.backward()
    with compiler_config.patch(activation_memory_budget=budget):
        recomputed = (compile_hyformer_layers(candidate.cuda()) if split_compile
                      else torch.compile(candidate.cuda(), backend="inductor"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = recomputed(*cloned)
            loss = actual[1].float().square().mean()
        loss.backward()
    pairs = list(zip(actual, expected))
    pairs.extend((a.grad, b.grad) for a, b in zip(candidate.parameters(), baseline.parameters())
                 if b.grad is not None)
    pairs.extend((a.grad, b.grad) for a, b in zip(cloned, tensors) if b.requires_grad)
    max_absolute_error = 0.0
    for actual_tensor, expected_tensor in pairs:
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.016, atol=1e-5)
        max_absolute_error = max(max_absolute_error,
                                 (actual_tensor.float() - expected_tensor.float()).abs().max().item())
    print("SISA_MEMORY_EQUIVALENCE " + json.dumps(dict(
        rank=int(os.environ.get("RANK", 0)), tensors_compared=len(pairs),
        dtype="BF16 autocast", max_absolute_error=max_absolute_error,
        rtol=0.016, atol=1e-5, activation_memory_budget=budget,
        split_hyformer_compile=split_compile, passed=True)), flush=True)
    del compiled, recomputed, baseline, candidate, actual, expected, tensors, cloned, pairs
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main() -> None:
    options, forwarded = parse_arguments(sys.argv[1:])
    import torch
    import torch.distributed as dist
    import torch._functorch.config as compiler_config

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if options.verify_equivalence:
        verify_equivalence(options.activation_memory_budget, options.split_hyformer_compile)
    compiler_config.activation_memory_budget = options.activation_memory_budget
    if options.split_hyformer_compile:
        install_layerwise_compilation()
    print("SISA_COMPILER_MEMORY_POLICY " + json.dumps(dict(
        rank=int(os.environ.get("RANK", 0)), activation_memory_budget=options.activation_memory_budget,
        torch_version=str(torch.__version__), probe_steps=options.probe_steps,
        split_hyformer_compile=options.split_hyformer_compile)), flush=True)
    if options.probe_steps:
        original_step = BaseModel.train_step
        completed_steps = 0

        def probe_step(model, batch):
            nonlocal completed_steps
            loss = original_step(model, batch)
            if not torch.isfinite(loss).all():
                raise ValueError("Nonfinite probe loss")
            completed_steps += 1
            if completed_steps == options.probe_steps:
                torch.cuda.synchronize()
                print("SISA_MEMORY_PROBE_PASS " + json.dumps(dict(
                    rank=model.rank, steps=completed_steps, loss=loss.item(),
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                    activation_memory_budget=options.activation_memory_budget,
                    split_hyformer_compile=options.split_hyformer_compile)), flush=True)
                if dist.is_initialized():
                    dist.barrier(device_ids=[local_rank])
                raise SystemExit(0)
            return loss

        BaseModel.train_step = probe_step
    sys.argv = ["run_expid.py", *forwarded]
    runpy.run_path(str(ROOT / "run_expid.py"), run_name="__main__")


if __name__ == "__main__":
    main()
