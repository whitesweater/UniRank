#!/usr/bin/env python3
"""Audit the strict OneTrans-Taobao baseline calibration artifacts."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


REFERENCE_AUC = {
    "is_click": 0.629531,
    "cart": 0.745796,
    "fav": 0.783280,
    "buy": 0.767286,
}
METRIC_PATTERN = re.compile(
    r"\[Task:\s*([^\]]+)\]\[Metrics\].*?AUC:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
ERROR_PATTERN = re.compile(
    r"Traceback|CUDA out of memory|RuntimeError|"
    r"(?:^|[^A-Za-z0-9_])(?:nan|inf)(?:[^A-Za-z0-9_]|$)",
    flags=re.IGNORECASE | re.MULTILINE,
)


def extract_test_auc(checkpoint_text: str) -> dict[str, float]:
    marker = "******** Test evaluation ********"
    if marker not in checkpoint_text:
        raise ValueError("test-evaluation marker is missing")
    test_section = checkpoint_text.rsplit(marker, 1)[1]
    metrics = {
        task.strip(): float(value)
        for task, value in METRIC_PATTERN.findall(test_section)
    }
    missing = sorted(set(REFERENCE_AUC) - set(metrics))
    if missing:
        raise ValueError(f"test AUC is missing for tasks: {missing}")
    if not all(math.isfinite(metrics[task]) for task in REFERENCE_AUC):
        raise ValueError("non-finite test AUC found")
    return {task: metrics[task] for task in REFERENCE_AUC}


def validate_protocol(slurm_text: str, checkpoint_text: str) -> list[str]:
    required_slurm = (
        "STRICT_PROTOCOL world_size=4 per_gpu_batch=8192 global_batch=32768 "
        "accumulation_steps=1 epochs=1 seed=20262027 bf16=true",
        "setting=baseline",
        "SISA_NATIVE_STRICT_COMPLETE",
    )
    required_checkpoint = (
        '"batch_size": "8192"',
        '"accumulation_steps": "1"',
        '"epochs": "1"',
        '"seed": "20262027"',
        '"max_len": "100"',
        '"world_size": "4"',
        '"enable_bf16": "True"',
        "Start training: 540 local batches/epoch",
        "DDP blocked training: True",
    )
    errors = [
        f"Slurm log missing: {needle}"
        for needle in required_slurm
        if needle not in slurm_text
    ]
    errors.extend(
        f"checkpoint log missing: {needle}"
        for needle in required_checkpoint
        if needle not in checkpoint_text
    )
    if '"sisa_enabled": "True"' in checkpoint_text:
        errors.append("baseline calibration unexpectedly enabled SISA")
    if ERROR_PATTERN.search(slurm_text) or ERROR_PATTERN.search(checkpoint_text):
        errors.append("an error/non-finite pattern was found")
    return errors


def build_report(
    metrics: dict[str, float],
    tolerance: float,
) -> tuple[str, bool]:
    rows = [
        "| Task | Official AUC | Reproduced AUC | Delta | Within reference tolerance |",
        "|---|---:|---:|---:|:---:|",
    ]
    within_all = True
    for task, reference in REFERENCE_AUC.items():
        reproduced = metrics[task]
        delta = reproduced - reference
        within = abs(delta) <= tolerance
        within_all = within_all and within
        rows.append(
            f"| {task} | {reference:.6f} | {reproduced:.6f} | "
            f"{delta:+.6f} | {'yes' if within else 'review'} |"
        )
    return "\n".join(rows), within_all


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slurm-log", required=True, type=Path)
    parser.add_argument("--checkpoint-log", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()

    slurm_text = args.slurm_log.read_text(encoding="utf-8", errors="replace")
    checkpoint_text = args.checkpoint_log.read_text(
        encoding="utf-8", errors="replace"
    )
    protocol_errors = validate_protocol(slurm_text, checkpoint_text)
    try:
        metrics = extract_test_auc(checkpoint_text)
    except ValueError as error:
        protocol_errors.append(str(error))
        metrics = {}

    if protocol_errors:
        print("STRICT CALIBRATION STRUCTURAL AUDIT: FAIL")
        for error in protocol_errors:
            print(f"- {error}")
        return 2

    report, within_all = build_report(metrics, args.tolerance)
    print("STRICT CALIBRATION STRUCTURAL AUDIT: PASS")
    print(report)
    if within_all:
        print("Reference comparison: all AUC deltas are within the diagnostic tolerance.")
    else:
        print(
            "Reference comparison: at least one AUC delta exceeds the diagnostic "
            "tolerance; review code/environment evidence, but this is not a "
            "structural protocol failure."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
