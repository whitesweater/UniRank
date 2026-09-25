"""Run HiFormer target-only RISE once on MerRec, KuaiRand and TencentGR.

Reuse the audited seed20262028 runner in a private module namespace. Historical
runner files and frozen QK experiments remain unchanged; only the task scope,
entry point, artifact location and provenance manifest are extended.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ENTRY = Path(__file__).resolve()
SBATCH = ROOT / "scripts/submit_hiformer_target_only_extension.sbatch"
LEGACY_RUNNER = ROOT / "experiments/rise_sample_analysis/scripts/run_targeted_controls.py"
ARTIFACTS = ROOT / "artifacts/rise_hiformer_target_only_extension"
TASKS = (
    {"model": "HiFormer", "dataset": "MerRec_Action", "mode": "target_only",
     "reference_task": "548166_2", "expected_dense": 67_481_031},
    {"model": "HiFormer", "dataset": "KuaiRand_Video_Action", "mode": "target_only",
     "reference_task": "548166_6", "expected_dense": 97_014_120},
    {"model": "HiFormer", "dataset": "TencentGR_10M_Action", "mode": "target_only",
     "reference_task": "548166_14", "expected_dense": 87_154_804},
)

_SPEC = importlib.util.spec_from_file_location(
    f"{__name__}._audited_targeted_runner", LEGACY_RUNNER)
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)
_LEGACY_SOURCE_HASHES = _BASE.source_hashes
_LEGACY_REVIEW_MANIFEST = _BASE.review_manifest


def identifier(task_id, attempt):
    if task_id not in range(len(TASKS)) or attempt < 1:
        raise ValueError("Invalid HiFormer extension task or attempt")
    return (f"RISE_hiformer_target_only_{TASKS[task_id]['dataset']}"
            f"_seed20262028_attempt{attempt}")


def training_command(task_id, attempt):
    identifier(task_id, attempt)
    return [str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nproc_per_node=2",
            str(ENTRY), "--worker", "--task-id", str(task_id), "--attempt", str(attempt)]


def source_hashes():
    sources = _LEGACY_SOURCE_HASHES()
    for path in (ENTRY, SBATCH):
        sources[str(path.relative_to(ROOT))] = _BASE.sha256(path)
    return sources


def review_manifest(task_id, attempt):
    identifier(task_id, attempt)
    manifest = _LEGACY_REVIEW_MANIFEST(task_id, attempt)
    manifest.update(
        study="rise_hiformer_target_only_extension",
        scope=("Three additional HiFormer datasets, target-only RISE, seed20262028; "
               "same historical full-RISE budget and initialization contract. "
               "QK is excluded because its target-only result already exists. "
               "This placement comparison changes active parameter count."),
        construction_contract={
            "enabled_target_sites": 1,
            "enabled_field_sites": 0,
            "target_parameter_seed": 20260822,
            "disabled_sites": "None; no inactive field adapter tensors",
        },
    )
    return manifest


# The legacy functions resolve globals in this private module, including the
# worker's source/config checks. Every subprocess therefore uses the new scope.
_BASE.TASKS = TASKS
_BASE.ARTIFACTS = ARTIFACTS
_BASE.identifier = identifier
_BASE.training_command = training_command
_BASE.source_hashes = source_hashes
_BASE.review_manifest = review_manifest
reference_params = _BASE.reference_params
effective_params = _BASE.effective_params
expected_counts = _BASE.expected_counts
prepare = _BASE.prepare


def main():
    return _BASE.main()


if __name__ == "__main__":
    raise SystemExit(main())
