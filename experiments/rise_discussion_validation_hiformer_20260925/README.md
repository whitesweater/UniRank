# Independent HiFormer replication

This replication was selected from read-only input coverage before inspecting
the primary RankMixer experiment's results. It uses the historical FULL-SITE
HiFormer--TencentGR checkpoint, seed 20262028, and the original eight 16384-row
compiled BF16 diagnostic windows. It is not the current history-only variant.
No training or model-source changes are made.

All 131072 rows remain in their original forward batches. The common comparison
set contains 131070 rows with at least two allowed keys and same-mask peers in
another physical part. The two excluded rows (empty history, one-key history)
and their physical identifiers are frozen in `plan.json`; exclusion is defined
from masks and parts without using outcomes. Raw-template, RMS-template and
donor routes preserve full correction at those two rows and all masked keys.
Zero-correction inference still runs all rows; its estimates use the same
common comparison set.

For each native mask, estimate a mean correction using only other physical
parts. The raw control retains the centered mean template's original scale;
the RMS control rescales that shape to the recipient's centered RMS. Both add
the recipient's original allowed-key rowmean. Three fixed-seed donor controls
sample with replacement from same-mask rows in other parts, then match the
recipient's centered RMS and rowmean. These maps are not bijections. Labels
never enter template or donor construction.

The primary endpoint is mean click BCE(full) minus BCE(template_rms) on the
131070 eligible rows. Conversion, equal-two-label macro loss, donor-averaged
per-row losses, and signed off-minus-raw / raw-minus-RMS / RMS-minus-full /
full-minus-off contrasts are secondary. Use 2000 paired user-cluster bootstrap
draws, seed 20260925, with interaction weighting and user-balanced click
sensitivity. Report the three donor seed estimates separately as a range.

Require bitwise historical512 plain prediction/raw reconstruction agreement,
all-window identity-hook agreement with a fresh plain pass, and unchanged
all-row QKV, native masks, scale and native scores for every intervention.
There are no historical all-window or historical local-off outputs to compare,
so those checks are not claimed. Compare actual BF16 replacement biases with
the intended one-cast tensor exactly and record centered RMS/mean roundoff.
Attention arrays are FP32 softmax reconstructions from captured native BF16
Q/K/Delta, not the internal fused kernel's captured probabilities. Invalid
empty-row attention diagnostics remain undefined and are explicitly flagged.

Freeze `plan.json` before any new GPU inference. All scripts, checkpoint,
historical source files and cache hashes are checked before and after running.
No full Q/K/V arrays are persisted; fixed row chunks are hashed in memory.
These are frozen-window diagnostic estimates, not full-test performance.
