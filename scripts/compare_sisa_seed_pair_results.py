#!/usr/bin/env python3
"""Compare two completed SISA single-seed studies with the same task matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_sisa_single_seed_results import (
    build_headline,
    compare_metrics,
    read_csv,
    summarize_dimension,
    summarize_rankings,
    summarize_units,
    write_csv,
    write_summary,
    DATASETS,
    MODELS,
)


def seed_pair_paths(
    reference_seed: int,
    candidate_seed: int,
) -> tuple[Path, Path, Path]:
    for name, value in (
        ("reference seed", reference_seed),
        ("candidate seed", candidate_seed),
    ):
        if not 0 <= value < 2**32 - 1:
            raise ValueError(f"{name} must be in [0, {2**32 - 2}], got {value}")
    if reference_seed == candidate_seed:
        raise ValueError("reference and candidate seeds must differ")
    reference = Path(
        f"experiments/sisa_single_seed{reference_seed}/results/metrics.csv"
    )
    candidate = Path(
        f"experiments/sisa_single_seed{candidate_seed}/results/metrics.csv"
    )
    output = Path(
        f"experiments/sisa_single_seed{candidate_seed}/"
        f"comparison_vs_seed{reference_seed}"
    )
    return reference, candidate, output


def validate_study_rows(
    rows: list[dict[str, str]],
    expected_seed: int,
    source: str,
) -> None:
    if len(rows) != 68:
        raise ValueError(f"{source} must contain 68 label rows, found {len(rows)}")
    observed_seeds = {int(row["seed"]) for row in rows}
    if observed_seeds != {expected_seed}:
        raise ValueError(
            f"{source} seed mismatch: expected {expected_seed}, "
            f"found {sorted(observed_seeds)}"
        )
    protocols = {row["protocol"] for row in rows}
    if protocols != {"ws2_bs16384_acc1"}:
        raise ValueError(
            f"{source} protocol mismatch: expected ws2_bs16384_acc1, "
            f"found {sorted(protocols)}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--reference-seed", type=int, required=True)
    parser.add_argument("--candidate-seed", type=int, required=True)
    args = parser.parse_args(argv)
    repository = args.repository.resolve()

    try:
        reference_path, candidate_path, output_path = seed_pair_paths(
            args.reference_seed,
            args.candidate_seed,
        )
        reference_rows = read_csv(repository / reference_path)
        candidate_rows = read_csv(repository / candidate_path)
        validate_study_rows(reference_rows, args.reference_seed, "reference metrics")
        validate_study_rows(candidate_rows, args.candidate_seed, "candidate metrics")
    except ValueError as error:
        parser.error(str(error))

    previous_rows = []
    for row in reference_rows:
        normalized = dict(row)
        normalized["previous_source"] = f"sisa_single_seed{args.reference_seed}"
        normalized["previous_protocol"] = row["protocol"]
        normalized["previous_gpu"] = row.get("gpu_name", "unknown")
        previous_rows.append(normalized)

    label_rows = compare_metrics(candidate_rows, previous_rows)
    unit_rows = summarize_units(label_rows)
    ranking_rows = summarize_rankings(label_rows)
    model_rows = summarize_dimension(unit_rows, label_rows, "model", MODELS)
    dataset_rows = summarize_dimension(unit_rows, label_rows, "dataset", DATASETS)
    headline = build_headline(label_rows, unit_rows, ranking_rows)
    headline.update(
        {
            "reference_seed": args.reference_seed,
            "candidate_seed": args.candidate_seed,
            "comparison_protocol": "ws2_bs16384_acc1",
        }
    )

    output_directory = repository / output_path
    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(output_directory / "label_deltas.csv", label_rows)
    write_csv(output_directory / "unit_summary.csv", unit_rows)
    write_csv(output_directory / "model_summary.csv", model_rows)
    write_csv(output_directory / "dataset_summary.csv", dataset_rows)
    write_csv(output_directory / "ranking_stability.csv", ranking_rows)
    (output_directory / "headline.json").write_text(
        json.dumps(headline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(
        output_directory / "summary.md",
        headline,
        model_rows,
        dataset_rows,
        comparison_note=(
            f"Candidate seed {args.candidate_seed} is compared with reference seed "
            f"{args.reference_seed} under the same 2×H100, world-size 2, per-GPU "
            "batch 16384, accumulation 1 protocol. The dataloader and SISA "
            "initialization random streams intentionally advance with the experiment "
            "seed, so these are descriptive two-point repeatability deltas, not a "
            "variance or significance estimate."
        ),
    )
    print(
        "SISA_SEED_PAIR_COMPARISON "
        f"reference={args.reference_seed} candidate={args.candidate_seed} "
        f"labels={len(label_rows)} cells={len(unit_rows)} "
        f"output_dir={output_directory}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
