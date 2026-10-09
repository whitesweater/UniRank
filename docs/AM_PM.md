# Paper AM/PM implementation

This release candidate starts at the upstream UniRank baseline `90bb99a`.
Its model change is an additional low-dimensional AM/PM attention-score branch:

```text
score = native_QK_score + AM_PM_score
output = native_attention_activation(score, native_mask) @ native_V
```

AM uses input-dependent cumulative attenuation. PM uses input-dependent
cumulative pairwise rotations. The existing Q/K/V projections, FFNs, residuals,
prediction towers, supervision, optimizers and data-loader behavior are retained.
The implementation is named SISA internally; the paper calls the method RISE,
and the supplementary component analysis calls phase modulation PA.

## Insertion sites

| Backbone | Enabled AM/PM sites | Retained scan semantics |
| --- | --- | --- |
| HiFormer | Target-to-history pooling only | History followed by the target |
| RankMixer | Target-to-history pooling | History followed by the target |
| Zenith | Target-to-history pooling only | History followed by the target |
| HyFormer | History sequence encoders and history-to-query decoders | Sequence encoding scans history; decoding scans history followed by global queries |
| OneTrans | Native mixed-attention layers | History followed by the native non-sequential tokens |

HiFormer and Zenith field/token interaction blocks are exactly the baseline
implementations. No process-local patching or experiment-directory imports are
needed to select these sites.

This is the previously validated AM/PM computation with selected insertion
sites. It does not introduce a new behavior-only state clock: target, global
query or non-sequential tokens still participate where the original adapter
included them. It also adds no timestamp features or temporal-bucket variants.

## Configuration

AM/PM is disabled by default. `sisa_enabled: False` constructs no auxiliary
parameters. `--rise-enabled` and `--sisa-enabled` are equivalent flags; the
corresponding `--no-rise-enabled` flag selects the native backbone.

The paper component configurations remain available:

| `sisa_variant` | Auxiliary content | AM | PM |
| --- | --- | --- | --- |
| `full` | Yes | Yes | Yes |
| `content_only` | Yes | No | No |
| `decay_only` | Yes | Yes | No |
| `rotation_only` | Yes | No | Yes |

Native baseline means disabling the entire branch; `content_only` still adds
auxiliary content projections. Default per-head channel dimension is 16 and
initial head scale is 0.1. Auxiliary initialization uses an isolated generator;
set `--sisa-parameter-seed` explicitly to reproduce a historical run's auxiliary
initialization. Training seed and batch configuration remain in the original
model YAML. `--run-id` separates logs and checkpoint names without changing
the training budget.

## Running and verification

Use `uv sync --locked` to create the pinned Python 3.12 / PyTorch 2.8 environment.
Place the upstream processed data and its feature-map metadata under `datasets/`
or edit the dataset paths for your deployment. These local paths are the only
changes to the baseline dataset configuration; labels and splits are retained.

From the repository root, run the native counterpart with the same experiment
and training parameters:

```bash
.venv/bin/torchrun --standalone --nproc_per_node=4 run_expid.py \
  --expid HiFormer_QK_Video_Action --gpu 0,1,2,3 \
  --no-rise-enabled --run-id HiFormer_QK_native

.venv/bin/torchrun --standalone --nproc_per_node=4 run_expid.py \
  --expid HiFormer_QK_Video_Action --gpu 0,1,2,3 \
  --rise-enabled --run-id HiFormer_QK_rise

.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_expid.py --help
```

CPU tests cover the scoring components, allowed insertion sites, preserved
backbone initialization, native-forward recovery, masks and gradients. They do
not establish four-GPU training performance. This branch contains no new
training results. Historical full-site HiFormer/Zenith checkpoints contain
additional field-adapter weights; use matching selective-site checkpoints for
this release's architecture.

Research reports, diagnostic controls, plotting, cluster scheduling and frozen
run records remain on the research branch. The upstream baseline documentation,
reference benchmarks and license are retained.
