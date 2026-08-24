# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import os
import sys
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import random
import inspect


def disable_torch_compile(fn):
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "disable"):
        return compiler.disable(fn)
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "disable"):
        return dynamo.disable(fn)
    return fn


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def build_batch_dict_from_tensor(batch_tensor, batch_columns):
    """Slice a batch tensor into a feature dictionary."""
    return {column: batch_tensor[:, index] for column, index in batch_columns}


def parse_gpu_ids(gpu_arg):
    """Parse a comma-separated GPU list; -1 selects CPU."""
    value = str(gpu_arg).strip()
    if value == "-1":
        return []
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("--gpu cannot be empty; use --gpu -1 for CPU")
    if any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid --gpu value: {gpu_arg}; expected values such as 0,1,2,3")
    gpu_ids = [int(part) for part in parts]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"Duplicate GPU IDs in --gpu: {gpu_arg}")
    return gpu_ids


def setup_visible_devices(gpu_ids):
    """Set visible GPUs outside Slurm; preserve Slurm's allocation inside jobs."""
    if os.environ.get("SLURM_JOB_ID"):
        slurm_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if gpu_ids and not slurm_visible:
            raise RuntimeError(
                "Slurm GPU job has no CUDA_VISIBLE_DEVICES allocation"
            )
        visible_count = len(
            [device for device in (slurm_visible or "").split(",") if device]
        )
        if any(gpu_id >= visible_count for gpu_id in gpu_ids):
            raise ValueError(
                "Inside Slurm, --gpu uses logical allocation indices; "
                f"received {gpu_ids} for {visible_count} visible device(s)"
            )
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))


def init_distributed_env():
    """Read the distributed process coordinates injected by torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    return world_size > 1, rank, local_rank, world_size, local_world_size


def is_main_process(rank):
    return rank == 0


def distributed_barrier(local_rank=None):
    """Synchronize the active process group, using the NCCL device when required."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    if dist.get_backend() == "nccl" and local_rank is not None:
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()

def get_device(gpu=-1):
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")
    return device

def get_optimizer(
        optimizer,
        params,
        lr,
        weight_decay=0,
        optimizer_kwargs=None):
    params = list(params)
    if len(params) == 0:
        return None
    if isinstance(optimizer, str):
        optimizer_name = optimizer.strip()
        optimizer_key = optimizer_name.lower()
        heavyball_alias = {
            "muon": "Muon",
            "laprop": "LaProp",
            "scion": "Scion",
            "soap": "SOAP",
        }
        use_heavyball = optimizer_key.startswith("heavyball.")
        if use_heavyball:
            heavyball_name = optimizer_name.split(".", 1)[1]
        else:
            heavyball_name = heavyball_alias.get(optimizer_key)
            use_heavyball = heavyball_name is not None

        if use_heavyball:
            if sys.version_info < (3, 10):
                raise RuntimeError(
                    "HeavyBall 3.2.0 requires Python 3.10 or newer."
                )
            try:
                import heavyball
            except ImportError as error:
                raise ImportError(
                    "HeavyBall optimizer '{}' requires the 'heavyball' package. "
                    "Install dependencies with `pip install -r requirements.txt`."
                    .format(optimizer_name)
                ) from error

            heavyball_classes = {}
            for name in dir(heavyball):
                candidate = getattr(heavyball, name)
                if (
                    inspect.isclass(candidate)
                    and issubclass(candidate, torch.optim.Optimizer)
                ):
                    heavyball_classes[name.lower()] = name
            resolved_name = heavyball_classes.get(heavyball_name.lower())
            if resolved_name is None:
                raise NotImplementedError(
                    "HeavyBall optimizer={} is not supported.".format(
                        heavyball_name
                    )
                )
            optimizer_cls = getattr(heavyball, resolved_name)
        else:
            optimizer_alias = {
                "adam": "Adam",
                "adamw": "AdamW",
                "adagrad": "Adagrad",
                "sgd": "SGD",
                "sparseadam": "SparseAdam",
            }
            optimizer_name = optimizer_alias.get(optimizer_key, optimizer_name)
            try:
                optimizer_cls = getattr(torch.optim, optimizer_name)
            except AttributeError as error:
                raise NotImplementedError(
                    "optimizer={} is not supported.".format(optimizer_name)
                ) from error
    else:
        optimizer_cls = optimizer

    if not callable(optimizer_cls):
        raise NotImplementedError(
            "optimizer={} is not supported.".format(optimizer_cls)
        )

    kwargs = {"lr": lr}
    if "weight_decay" in inspect.signature(optimizer_cls).parameters:
        kwargs["weight_decay"] = weight_decay
    if optimizer_kwargs:
        kwargs.update(optimizer_kwargs)
    try:
        return optimizer_cls(params, **kwargs)
    except Exception as error:
        raise RuntimeError(
            "Failed to initialize optimizer={} with lr={} and weight_decay={}."
            .format(
                getattr(optimizer_cls, "__name__", str(optimizer_cls)),
                lr,
                weight_decay,
            )
        ) from error


@torch.no_grad()
def chunked_adagrad_step(optimizer, chunk_size):
    """Apply the dense Adagrad formula with bounded temporary memory.

    PyTorch's single-tensor implementation materializes ``sqrt(state_sum)``
    for an entire parameter. A unified embedding table can make that
    temporary several GiB even though Adagrad is elementwise. Chunking the
    final division preserves the update while bounding the temporary tensor.
    """
    if not isinstance(optimizer, torch.optim.Adagrad):
        raise TypeError("chunked_adagrad_step requires torch.optim.Adagrad")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    for group in optimizer.param_groups:
        if group.get("differentiable", False):
            raise RuntimeError("chunked Adagrad does not support differentiable=True")
        if group.get("fused", False):
            raise RuntimeError("chunked Adagrad does not support fused=True")
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            if gradient.is_sparse:
                raise RuntimeError("chunked Adagrad requires dense gradients")
            if group.get("maximize", False):
                gradient = -gradient
            weight_decay = group.get("weight_decay", 0)
            if weight_decay != 0:
                gradient = gradient.add(parameter, alpha=weight_decay)

            state = optimizer.state[parameter]
            state["step"] += 1
            step = state["step"].item()
            state_sum = state["sum"]
            clr = group["lr"] / (1 + (step - 1) * group["lr_decay"])

            if torch.is_complex(parameter):
                parameter_view = torch.view_as_real(parameter)
                gradient_view = torch.view_as_real(gradient)
                state_sum_view = torch.view_as_real(state_sum)
            else:
                parameter_view = parameter
                gradient_view = gradient
                state_sum_view = state_sum

            state_sum_view.addcmul_(gradient_view, gradient_view, value=1)
            parameter_flat = parameter_view.reshape(-1)
            gradient_flat = gradient_view.reshape(-1)
            state_sum_flat = state_sum_view.reshape(-1)
            for start in range(0, parameter_flat.numel(), chunk_size):
                stop = min(start + chunk_size, parameter_flat.numel())
                standard_deviation = state_sum_flat[start:stop].sqrt().add_(
                    group["eps"]
                )
                parameter_flat[start:stop].addcdiv_(
                    gradient_flat[start:stop],
                    standard_deviation,
                    value=-clr,
                )

def get_loss(loss):
    if isinstance(loss, str):
        if loss in ["bce", "binary_crossentropy", "binary_cross_entropy"]:
            loss = "binary_cross_entropy"
    try:
        loss_fn = getattr(torch.functional.F, loss)
    except:
        try:
            loss_fn = eval("losses." + loss)
        except:
            raise NotImplementedError("loss={} is not supported.".format(loss))
    return loss_fn

def get_activation(activation, hidden_units=None):
    if isinstance(activation, str):
        activation_name = activation.strip().lower()
        if activation_name in ["prelu", "dice"]:
            assert type(hidden_units) == int
        if activation_name == "none":
            return None
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "softplus":
            return nn.Softplus()
        elif activation_name == "silu":
            return nn.SiLU()
        elif activation_name == "gelu":
            return nn.GELU()
        elif activation_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_name == "mish":
            return nn.Mish()
        elif activation_name == "tanh":
            return nn.Tanh()
        elif activation_name == "softmax":
            return nn.Softmax(dim=-1)
        elif activation_name == "prelu":
            return nn.PReLU(hidden_units, init=0.1)
        elif activation_name == "dice":
            from unirank.pytorch.layers.activations import Dice
            return Dice(hidden_units)
        else:
            return getattr(nn, activation)()
    elif isinstance(activation, list):
        if hidden_units is not None:
            assert len(activation) == len(hidden_units)
            return [get_activation(act, units) for act, units in zip(activation, hidden_units)]
        else:
            return [get_activation(act) for act in activation]
    return activation

def get_initializer(initializer):
    if isinstance(initializer, str):
        try:
            initializer = eval(initializer)
        except:
            raise ValueError("initializer={} is not supported."\
                             .format(initializer))
    return initializer
