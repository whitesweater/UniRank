# Revision 3: preflight process boundary

Revision 2 passed all first-window identity/capture/replay/zero fidelity gates for
both models, then both processes crashed on the first full-stage unmodified forward:
RankMixer job 652347 raised SIGILL and HiFormer job 652346 raised SIGSEGV. These were
different nodes. No nonidentity permutation outcome was reached. The runtime cause
is unproven; repeated compiler/model lifecycle and dynamically reloaded hook code
are plausible contributors.

Revision 3 makes one execution change: the unchanged first-window `fidelity_v2.py`
runs in a fresh Python subprocess and must complete successfully before the main
process constructs its first full-stage model. Sampling, model/checkpoint, precision,
source checks, permutation maps, endpoints, numerical tolerances and strict gates
are unchanged. Both prior versions and their failure evidence are preserved.

There is no training, tolerance relaxation, automatic retry, or selection of a
favorable result. `submit_two_stage_v3.sbatch` requires review and frozen version-3
plans before root-task submission.
