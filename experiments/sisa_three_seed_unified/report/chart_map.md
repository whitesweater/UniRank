# Chart map

| Section | Question | Family / type | Fields | Supported claim | Palette |
|---|---|---|---|---|---|
| SISA vs baseline | 哪些 16-cell 单元贡献 AUC 变化？ | Comparison / horizontalBar | unit, delta_auc_vs_baseline_milli | 展示方向和幅度 | diverging, midpoint 0 |
| Paper delta vs variability | 论文增益是否伴随高 seed 波动？ | Relationship / scatter | delta_auc_vs_paper_milli, sisa_seed_range_auc_milli, dataset | 识别高增益/高波动单元 | categorical by dataset |

The first visual uses ranked bars because labels are long and signed. The second uses 16 same-grain observations, meeting the scatter sufficiency gate.
