# Paper benchmark source

`paper_table.csv` materializes the 68 matching rows from **Table 2,
“Comprehensive Benchmarking Results (Sequence Length = 100)”** in UniRank,
arXiv:2607.19987, source version downloaded on 2026-08-26.

- Paper: <https://arxiv.org/abs/2607.19987>
- Source archive: <https://arxiv.org/e-print/2607.19987>
- Local dataset mapping: `TAAC-25` in the paper corresponds to
  `TencentGR_10M_Action` in this repository.
- Values preserve the paper table's four-decimal precision. They are not
  replaced with the higher-precision author benchmark logs.
- The unified comparison contains only HiFormer, HyFormer, RankMixer, and
  Zenith on QK-Video, KuaiRand, TAAC-25/TencentGR, and MerRec.
