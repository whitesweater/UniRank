# Native history order and the RISE score channel

Status: staged inference-only diagnostic; no training or parameter updates.

This study asks whether the inspected native pooling operation is insensitive to
joint permutations of its historical event records, while its cumulative RISE
score channel can distinguish their arrangement. It uses the existing audited
RankMixer–KuaiRand and historical full-site HiFormer–TAAC-25 checkpoints and the
same eight preselected 16,384-row windows per case. These are conditional checkpoint
diagnostics, not newly trained models or full-test benchmark estimates.

## Four main routes

All routes keep the original native Q/K/V, mask, scale, candidate and downstream
network. The same pre-hook gathers the inputs to the inspected target site's
SISAScoreBias; the stock score computation runs once, and the same output hook
inverse-gathers its key axis to the original event identities.

1. `identity`: identity gather before and after the native score module.
2. `off`: the same path followed by `zeros_like` on this site's correction only.
3. `reverse`: reverse valid historical event embeddings; padding and candidate stay fixed.
4. `random`: one prespecified independent per-row valid-history shuffle.

The shuffle uses NumPy `default_rng(20260925 + window_index)`. Its mapping is frozen
before inference and saved with the outputs. This is a perturbation seed, not a
training replication. No rowmean, RMS, moment or sign matching is applied. The
native/data-provided order is not presumed to be chronological.

Primary outcomes are paired click BCE(reverse − identity) and BCE(random − identity)
on history length at least 2. Every original row is retained; all-row results and
all other labels are secondary. Off − identity describes this site's existing
predictive utility separately from sensitivity to arrangement. A permutation may
improve prediction; such outcomes must be retained and reported without relabeling.
An independent analysis computes 2,000 paired user-cluster bootstrap draws using
seed 20260925, plus user-balanced sensitivity.

## Fidelity and native numerical floor

Unmodified checkpoint predictions must first reproduce the original 512 attention
capture rows. The new identity hook must match fresh unmodified inference and the
earlier 131,072-row probe's full predictions bitwise. Captured original score data
must match the historical attention capture. The off route must match prior local-off
predictions bitwise. Every intervention must retain exact original Q/K/V, mask,
scale and native-score hashes. No outcome is accepted after a failed gate.

A separate 64-row real-input check jointly permutes every historical item attribute
and corresponding masks, leaving candidate, padding, nonhistory fields and labels
fixed. It disables only the inspected score correction. Shared-weight eager modules
with autocast disabled and the FP32 MATH backend must retain pooling within
`atol=1e-5, rtol=1e-4`. Prediction differences are recorded. Eager BF16 with automatic
SDPA selection is separately measured as a precision-related numerical residual;
it does not use the compiled main-route path. Bitwise permutation invariance is
not assumed. This small check does not enter the main BCE analysis.

CPU checks cover inverse alignment (including a content-only SISA control), irregular
padding, empty/singleton histories, fixed candidates, raw-record round trips and
deterministic shuffles.

## Locations and execution

New source/plans: `experiments/rise_history_permutation_20260925/`.
New outputs: `artifacts/rise_history_permutation_20260925/<case>/job_<id>/`.
Existing model sources, datasets, checkpoints and active jobs remain untouched.

Before any GPU run, review source and run:

```bash
.venv/bin/python experiments/rise_history_permutation_20260925/test_permutation.py
bash -n experiments/rise_history_permutation_20260925/submit_permutation.sbatch
.venv/bin/python experiments/rise_history_permutation_20260925/run_permutation.py --case rankmixer_kuairand --prepare
.venv/bin/python experiments/rise_history_permutation_20260925/run_permutation.py --case hiformer_taac25 --prepare
```

Prepare refuses to overwrite plans and verifies the earlier audited source/input
hashes. Plans freeze own source, previous probe references and every permutation
mapping hash. The launcher runs each case on one GPU, sequentially by default;
staging these files does not submit any job.
