# Repository Guidelines

## Project Structure & Module Organization

Core framework code lives in `unirank/`; model implementations and SISA adapters live in `model_zoo/`. Experiment and dataset settings are defined in `config/model_config.yaml` and `config/dataset_config.yaml`. Use `run_expid.py` for a single experiment or DDP run. Cluster submission, auditing, and result collection tools belong in `scripts/`. Tests are in `tests/` and follow the source behavior they protect.

Keep formal plans, reports, and compact result tables under `experiments/<study_slug>/`. `benchmark/` contains upstream reference logs used for baseline checks, while `data/` contains preprocessing programs. Runtime data and outputs belong in `datasets/`, `logs/`, `checkpoints/`, and `artifacts/`; these are intentionally excluded from Git. Documentation images live in `assets/figures/`.

## Build, Test, and Development Commands

```bash
uv sync --locked
.venv/bin/python -m unittest discover -s tests -v
bash -n scripts/*.sbatch
.venv/bin/python run_expid.py --help
```

Use `uv` as the sole environment and dependency manager. `uv sync --locked` recreates `.venv` from `pyproject.toml` and `uv.lock` with the pinned Python 3.12/PyTorch stack. Add or remove dependencies with `uv add` or `uv remove`, commit both metadata files, and do not use ad-hoc `pip install`. Run the full CPU regression suite before committing. Validate modified Slurm scripts with `bash -n`. For four-GPU training, use the repository launchers or `torchrun --nproc_per_node=4 run_expid.py ...`; do not simulate a formal result with a single-GPU command.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase constants. Keep imports grouped as standard library, third-party packages, then project modules. Match existing type hints and avoid new dependencies unless required. Experiment IDs use `Model_Dataset` patterns; run IDs should add the study, setting, and seed. Name Slurm files `submit_<study>.sbatch` and result tools with verbs such as `collect_` or `audit_`.

## Testing Guidelines

Tests use Python `unittest`; files must be named `test_*.py`, classes `*Test`, and methods `test_*`. Add a regression test before changing protocols, checkpoint retention, task mappings, or result schemas. Keep deterministic unit tests CPU-compatible; document GPU-only validation separately. All tests and relevant shell syntax checks must pass.

## Commit & Pull Request Guidelines

Recent history uses short, imperative commit subjects, for example `Consolidate experiment reports under experiments`. Keep each commit focused. Pull requests should describe the behavior or experiment protocol changed, list validation commands, identify affected configs and job/task IDs, and link the resulting report or compact CSV. Never commit datasets, full Slurm logs, `.model` files, credentials, or machine-specific paths.
