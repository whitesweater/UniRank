# Native capture and replay revision

The first implementation failed identity fidelity in jobs 652247 (RankMixer) and
652246 (HiFormer), before running either nonidentity permutation. Those outputs
are invalid as scientific evidence. Source, plans, gates, logs and accounting are
preserved under `artifacts/rise_history_permutation_20260925/failure_archive/array_652246/`.
The unmodified checkpoint predictions matched historical captures. The new internal
hooks changed numerical behavior; HiFormer preserved captured score data but changed
downstream predictions, while RankMixer also changed captured score data. These
observations motivate removing all new SISA and pooling hooks.

Revision 2 uses only the existing audited dot-attention Observer and replay code:

1. First-window engineering preflight verifies original predictions, native score
   capture, captured-score identity replay, and zero replay against the historical
   original/local-off outputs. All comparisons require bitwise fidelity.
2. Full-window original capture, identity replay and zero replay must pass before
   any reverse or shuffled history is evaluated.
3. Jointly permute all raw historical item attributes and masks **outside** the
   compiled model. Capture the correction produced by its unchanged native score
   module. Inverse-align the captured correction to the original key identities.
   Also inverse-align actual captured K/V outside forward and verify original
   content hashes; candidate query and mask must match exactly.
4. Replay this captured correction on the original records using the same audited
   replacement tensor strategy. Original Q/K/V, mask and scale must remain exact;
   actual injected BF16 correction must equal the intended capture.

There are no new SISA/pooling hooks in the main routes and no training. Main pooling
difference columns are removed; attention TV and prediction/correction differences
remain. The separate small eager FP32/BF16 native numerical check retains its
previously declared scope. Sampling, labels, masks, permutation maps, length>=2
eligibility, BCE endpoints and paired user-cluster analysis are unchanged.

The first-window preflight and full diagnostic run share one allocation; execution
proceeds only when the strict gates pass. Plans and this revision must be reviewed
and frozen before launch. The old source and old plans remain unmodified.

```bash
.venv/bin/python experiments/rise_history_permutation_20260925/run_two_stage_v2.py --case rankmixer_kuairand --prepare
.venv/bin/python experiments/rise_history_permutation_20260925/run_two_stage_v2.py --case hiformer_taac25 --prepare
```

After review, the root task can submit `submit_two_stage_v2.sbatch`. No automatic
resubmission is performed by these scripts.
