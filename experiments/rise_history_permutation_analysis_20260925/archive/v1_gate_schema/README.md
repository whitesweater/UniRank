# Independent history-permutation analysis

This analysis concerns frozen RankMixer/HiFormer checkpoints only. No adaptation,
training or GPU work occurs here. The analysis source is outside the executor's
frozen runner directory.

Primary population: `effective_length >= 2`, determined solely from input metadata.
Both reverse-minus-identity and fixed-random-minus-identity click BCE differences
are primary endpoints. All 131,072 rows, every label, equal-weight label macro,
off-minus-identity utility and user-balanced statistics remain visible.

The formulae reproduce the previous score-probe convention: clip probabilities
to [1e-7, 1 - 1e-7], aggregate paired losses within users, draw 2,000 multinomial
user resamples using seed 20260925, and use 2.5/97.5 percentile intervals. The
row-weighted denominator changes with sampled user multiplicities. These are
conditional fixed-checkpoint/window/permutation intervals, not training or
random-permutation-distribution uncertainty.

Expected runner files are `complete.json`, `plan.json`, `gates.json`, `meta.csv`
and `predictions.csv`; the final completion hash manifest must cover all of them.
The prediction table retains source metadata and labels and provides columns
`<label>_identity`, `<label>_off`, `<label>_reverse`, `<label>_random`.

```text
python analyze_permutation.py /path/to/new/results --case rankmixer_kuairand --historical-results /path/to/previous/score/probe --expected-plan-sha256 FROZEN_SHA --output /path/to/new/analysis
```

Repeat for `hiformer_taac25`. The historical comparator uses prior 131,072-row
`full` and `local_off` predictions, not merely the earlier 512-row raw-capture
case. It independently checks identity metadata and reports every historical
prediction discrepancy. Default maximum absolute historical tolerance is 1e-5,
fixed before outcomes. Any change must be documented before results are inspected.

`analyze_permutation.sbatch` uses four CPUs, 16 GB and a 15-minute cap, with no
GPU request. It takes five arguments: case, new results, historical results,
expected frozen plan SHA-256, and a new output directory. It is prepared only;
the root agent controls submission after review and completed runner artifacts.

Outputs are small audit/summary JSON files, all metric/contrast CSV files,
per-user loss sums sufficient to reproduce the bootstrap, prediction-change
summaries, and exact row IDs whose new length-based eligibility differs from the
prior peer-based eligibility. No unfavorable contrast is filtered.

Raw-data integrity is evaluated here; the runner's identity-versus-plain,
unchanged-QKV and native-pooling joint-permutation gates additionally require
independent implementation/runtime review. The numerical floor uses 64 rows,
FP32 MATH and BF16 eager modules with automatic SDPA; it does not establish
compiled-path bitwise invariance. `off` disables only the inspected target-site
correction; other RISE sites in the checkpoint remain active. Sensitivity and utility must not be
collapsed into a claim that all gain is caused by order modeling.
