| Model | Seed | Full BCE | Shared raw BCE | Shared RMS BCE | Full - RMS [95% CI] |
|---|---:|---:|---:|---:|---:|
| RankMixer | 20262028 | 0.510378980 | 0.510388693 | 0.510341991 | +0.000036990 [-0.000114081, +0.000169580] |
| RankMixer | 20262029 | 0.509952468 | 0.510228935 | 0.510270094 | -0.000317627 [-0.000499123, -0.000145279] |

Two available checkpoints per model evaluated on the same fixed windows; each checkpoint gets its own recalculated profiles and conditional 2000-draw user-cluster interval. No pooling of users across seeds, no three-seed claim, no training-randomness interval.
