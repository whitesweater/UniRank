#!/usr/bin/env python3
"""Four-rank NCCL smoke used by the HPC3/ACD preflight job."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    value = torch.tensor([float(local_rank + 1)], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    if value.item() != 10.0:
        raise SystemExit(
            f"ACD_PREFLIGHT_ERROR NCCL all_reduce returned {value.item()}"
        )
    if local_rank == 0:
        print("SISA_ACD_NCCL_COMPLETE world_size=4 all_reduce=10")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
