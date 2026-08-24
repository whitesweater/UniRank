#!/usr/bin/env python3
"""Run one real UniRank train step against a preprocessed local dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import model_zoo
from unirank.features import feature_map_from_config
from unirank.pytorch.dataloaders import RankDataLoader
from unirank.pytorch.layers import SISAScoreBias
from unirank.pytorch.torch_utils import seed_everything
from unirank.utils import load_config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config")
    parser.add_argument("--expid", default="RankMixer_Taobao_Action")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--sisa-enabled", action="store_true")
    parser.add_argument("--sisa-score-dim", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; submit this smoke test through Slurm.")

    params = load_config(args.config, args.expid)
    params.update(
        {
            "gpu": 0,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "num_workers": 0,
            "shuffle": False,
            "enable_torch_compile": False,
            "distributed": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "sisa_enabled": args.sisa_enabled,
            "sisa_score_dim": args.sisa_score_dim,
        }
    )
    seed_everything(params["seed"])

    dataset_dir = Path(params["data_root"]) / params["dataset_id"]
    required_paths = [
        Path(params[key])
        for key in ("train_data", "train_user_info", "train_item_info")
    ]
    missing = [str(path) for path in required_paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing preprocessed data paths: {missing}")

    feature_map = feature_map_from_config(params)
    model_class = getattr(model_zoo, params["model"])
    model = model_class(feature_map, **params)
    sisa_modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, SISAScoreBias)
    }
    if args.sisa_enabled and not sisa_modules:
        raise RuntimeError("SISA was enabled but the model has no SISA attention site")
    if not args.sisa_enabled and sisa_modules:
        raise RuntimeError("Baseline smoke unexpectedly constructed SISA parameters")
    lambda_before = {
        name: module.lambdas(dtype=torch.float32, device=model.device)
        .detach()
        .cpu()
        .tolist()
        for name, module in sisa_modules.items()
    }
    sisa_state_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "sisa_score_bias" in name
    }
    lambda_gradients = {}
    gradient_hooks = []
    for name, module in sisa_modules.items():
        gradient_hooks.append(
            module.raw_head_scale.register_hook(
                lambda gradient, parameter_name=name: lambda_gradients.__setitem__(
                    parameter_name,
                    gradient.detach().float().cpu().clone(),
                )
            )
        )

    train_gen, _ = RankDataLoader(
        feature_map, stage="train", **params
    ).make_iterator()
    batch = next(iter(train_gen))

    model.train()
    model._batch_index = 0
    model._max_gradient_norm = 10.0
    loss = model.train_step(batch)
    for hook in gradient_hooks:
        hook.remove()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite training loss: {loss.item()}")
    lambda_after = {
        name: module.lambdas(dtype=torch.float32, device=model.device)
        .detach()
        .cpu()
        .tolist()
        for name, module in sisa_modules.items()
    }
    lambda_changed = any(
        before != lambda_after[name]
        for name, before in lambda_before.items()
    )
    sisa_parameter_changed = any(
        not torch.equal(before, dict(model.named_parameters())[name].detach().cpu())
        for name, before in sisa_state_before.items()
    )
    lambda_gradient_norms = {
        name: float(gradient.norm())
        for name, gradient in lambda_gradients.items()
    }
    lambda_has_gradient = (
        len(lambda_gradient_norms) == len(sisa_modules)
        and all(
            torch.isfinite(torch.tensor(norm)) and norm > 0.0
            for norm in lambda_gradient_norms.values()
        )
    )
    if args.sisa_enabled and (
        not lambda_has_gradient or not sisa_parameter_changed
    ):
        print(
            "SISA_UPDATE_DIAGNOSTIC="
            + json.dumps(
                {
                    "experiment": args.expid,
                    "lambda_before": lambda_before,
                    "lambda_after": lambda_after,
                    "lambda_gradient_norms": lambda_gradient_norms,
                    "lambda_has_gradient": lambda_has_gradient,
                    "sisa_parameter_changed": sisa_parameter_changed,
                },
                sort_keys=True,
            )
        )
        raise RuntimeError("SISA learnability gate failed")

    model.eval()
    with torch.no_grad(), model._get_amp_context():
        predictions = model(batch)
    if not all(torch.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError("Model produced non-finite predictions.")

    summary = {
        "status": "ok",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model": params["model"],
        "experiment": args.expid,
        "dataset": params["dataset_id"],
        "dataset_dir": str(dataset_dir.resolve()),
        "batch_size": int(batch[2].shape[0]),
        "sequence_length": int(batch[2].shape[1]),
        "train_loss": float(loss.detach().cpu()),
        "prediction_keys": sorted(predictions),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "sisa_enabled": args.sisa_enabled,
        "sisa_attention_sites": sorted(sisa_modules),
        "sisa_lambda_before": lambda_before,
        "sisa_lambda_after": lambda_after,
        "sisa_lambda_changed": lambda_changed,
        "sisa_lambda_gradient_norms": lambda_gradient_norms,
        "sisa_lambda_has_gradient": lambda_has_gradient,
        "sisa_parameter_changed": sisa_parameter_changed,
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
    }
    print("UNIRANK_GPU_SMOKE=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
