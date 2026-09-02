#!/usr/bin/env python3
"""Canonical task mapping for the 16-cell SISA single-seed study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


MODELS = ("HiFormer", "HyFormer", "RankMixer", "Zenith")
DATASETS = (
    "QK_Video_Action",
    "KuaiRand_Video_Action",
    "TencentGR_10M_Action",
    "MerRec_Action",
)

# Put all MerRec and KuaiRand cells in the first wave, with the highest-memory
# Zenith-MerRec cell first, so resource failures surface early.
EXPERIMENTS = (
    "Zenith_MerRec_Action",
    "HyFormer_MerRec_Action",
    "HiFormer_MerRec_Action",
    "RankMixer_MerRec_Action",
    "Zenith_KuaiRand_Video_Action",
    "HyFormer_KuaiRand_Video_Action",
    "HiFormer_KuaiRand_Video_Action",
    "RankMixer_KuaiRand_Video_Action",
    "Zenith_QK_Video_Action",
    "HyFormer_QK_Video_Action",
    "HiFormer_QK_Video_Action",
    "RankMixer_QK_Video_Action",
    "Zenith_TencentGR_10M_Action",
    "HyFormer_TencentGR_10M_Action",
    "HiFormer_TencentGR_10M_Action",
    "RankMixer_TencentGR_10M_Action",
)


@dataclass(frozen=True)
class SingleSeedTask:
    task_id: int
    model: str
    dataset: str

    @property
    def experiment(self) -> str:
        return f"{self.model}_{self.dataset}"


def single_seed_task(task_id: int) -> SingleSeedTask:
    if not 0 <= task_id < len(EXPERIMENTS):
        raise ValueError(
            f"task id must be in [0, {len(EXPERIMENTS) - 1}], got {task_id}"
        )
    experiment = EXPERIMENTS[task_id]
    model, dataset = experiment.split("_", 1)
    return SingleSeedTask(task_id=task_id, model=model, dataset=dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    task = single_seed_task(args.task_id)
    print(f"{task.model}\t{task.dataset}")


if __name__ == "__main__":
    main()
