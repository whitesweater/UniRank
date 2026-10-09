| Model | Seed | Full BCE | Shared raw BCE | Shared RMS BCE | Full - RMS [95% CI] |
|---|---:|---:|---:|---:|---:|
| HiFormer | 20262028 | 0.045460577 | 0.045504469 | 0.045235255 | +0.000225322 [+0.000190107, +0.000261512] |
| HiFormer | 20262029 | 0.049485439 | 0.049103303 | 0.048836844 | +0.000648595 [+0.000606696, +0.000691587] |

Two available checkpoints per model evaluated on the same fixed windows; each checkpoint gets its own recalculated profiles and conditional 2000-draw user-cluster interval. No pooling of users across seeds, no three-seed claim, no training-randomness interval.
