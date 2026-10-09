# Independent AUC-results and report review

Date: 2026-09-26. Reviewer: `adapter_code_review`.

**Approved.** Job 652845 completed with exit 0:0. The saved numerical results and the final Chinese/English REPORT.md accurately describe the evidence. This review made no manuscript or task changes and writes only its own review files in the new statistics directory. It does not claim that concurrently edited manuscript files remained unchanged.

## Numerical verification

- Recomputed the frozen plan hash and independently verified all 6 completed-result files, 4 source files, and 28 input-file hashes. The plan hash is `e0ecb3a4221290dd1985abae296abf07ecc659a78ff18d2d923b59cc8919c414`.
- Confirmed saved AUC-array shapes of `10000 x 2 x 2 x 6` for RankMixer and `10000 x 2 x 2 x 2` for HiFormer, representing draws, checkpoint seeds, routes, and labels.
- Reconstructed the prescribed user-multiplicity streams and matched their hashes. Independently recomputed positive/negative class-presence masks for all 10,000 draws from the original labels. There are no invalid draws for any label or endpoint.
- Recomputed every macro endpoint within each draw and reproduced all eight nominal and adjusted interval bounds exactly from the saved arrays. The adjusted quantiles are 0.003125 and 0.996875. Every reported direction and valid-draw count matches.
- For fixed draws 0, 4999, and 9999, compared original predictions with sklearn ROC AUC using the reconstructed user sample weights. All 96 comparisons passed; the maximum discrepancy was 2.22e-16. The full AUC bootstrap was not rerun.
- Equivalent orders of macro averaging and subtraction differed by at most 4.08e-16, consistent with floating-point roundoff. The report's actual calculation order and quantiles reproduce exactly.

## Substantive conclusion

HiFormer seed 20262029 has positive click and macro AUC point differences, but both ordinary 95% intervals already cross zero. Neither it nor any other case establishes a macro-AUC direction under the eight-endpoint adjustment.

Only two adjusted click-AUC intervals exclude zero: RankMixer seed 20262029 favors Full, while HiFormer seed 20262028 favors RMS. The other six adjusted intervals cross zero. This is not evidence of equivalence, and the focal HiFormer seed 20262029 must not be described as having an established AUC advantage.

## REPORT.md review

Verified the displayed focal-case AUC values, both sets of intervals, percentage-point conversion, and the earlier BCE values against the retained outputs. The report accurately distinguishes Full versus RMS from RISE-on versus RISE-off and from independently trained baselines. It correctly retains recipient-specific RMS in the shared-profile definition and avoids implying unchanged downstream tensors.

The final standalone English paragraph explicitly includes 10,000 paired user-cluster bootstrap resamples, the eight-comparison Bonferroni family, and the equal-weight click/conversion macro definition. Both language versions state the post hoc exploratory status and the fixed-checkpoint/window/template uncertainty scope. No p-value, equivalence claim, calibration mechanism, or broad cross-seed advantage is invented.

Final reviewed REPORT.md SHA-256: `47f60922bbfea11fea8a5e20421e4c05d9aff362c505a7ac4d098ed41a8c8a9a`.

There are no remaining numerical or report-writing blockers.
