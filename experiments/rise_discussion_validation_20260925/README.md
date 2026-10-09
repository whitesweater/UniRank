# Frozen-window scoring intervention for Discussion

This experiment reuses all 131072 interactions in the eight original 16384-row
RankMixer--KuaiRand diagnostic windows. It preserves the trained checkpoint,
batch layout, compiled BF16 inference, original inputs, upstream Q/K/V, native
mask and scale. It does not retrain or modify model sources.

Prepare `plan.json` before GPU inference. The plan freezes the input and source
hashes, outcome-independent route definitions, all-row inclusion, original
sampling frame, primary click BCE comparison, secondary all-label and macro
BCE comparisons, donor seeds, and user-cluster uncertainty protocol.

Compare full correction against zero correction, a mean correction estimated
from the seven other physical parts, the same mean shape rescaled to the
recipient correction's centered RMS, and three seeded one-to-one donor maps
from different physical parts. All nonzero controls have the recipient's
original row mean before BF16 casting. The raw template preserves the cohort
centered RMS; the matched template and donors preserve recipient centered RMS.
No label is used to build any correction or donor assignment.

The primary endpoint is interaction-weighted mean click BCE(full) minus
BCE(template_rms). Report 2000 paired user-cluster bootstrap draws with fixed
seed 20260925, and user-balanced click sensitivity. Aggregate the three donor
losses within each interaction before uncertainty estimation. Secondary
contrasts are off minus raw, raw minus RMS, RMS minus full, full minus off;
report signed values irrespective of their direction. Separate each label,
the equal-label macro, long-view and the donor seed range. These are diagnostic
window estimates, not whole-test AUC or population estimates.

Numerical gates require bitwise identity with the original 512 saved predictions
and original all-window prediction hashes; identity-hook agreement on all rows;
original selected QKV hashes/raw arrays; unchanged all-row QKV/mask/scale on
every intervention; and historical local-off prediction agreement. Save actual
post-cast correction/attention arrays, cosine, TV, RMS and rounding errors.
No full Q/K/V arrays are persisted; row chunks are hashed in memory.

Run CPU checks: `.venv/bin/python -m unittest discover -s experiments/rise_discussion_validation_20260925 -p test_probe_math.py`.
Freeze plan: `.venv/bin/python experiments/rise_discussion_validation_20260925/run_probe.py --prepare`.
The submitter must inspect live Slurm partitions and provide an available H100
partition explicitly; the batch file intentionally does not guess a partition.
