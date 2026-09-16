# 消融实验索引

后续所有 SISA 或 UniRank 消融实验都放在本目录下，每个研究问题一个子目录，不按临时
Slurm job ID 建目录。

当前 study：`sisa_components_bcd`，RankMixer–KuaiRand 的 B/C/D 三组组件消融，
复用 H100 完整 SISA 对照。提交状态见该目录的 notes/launch.md。
追加 study：`sisa_components_hyformer_merrec_bcd`，HyFormer–MerRec 的 B/C/D 三组组件消融，
复用 H100 seed 20262028 完整 SISA 对照；具体记录位于该目录。
第三组：`sisa_components_hiformer_taac_bcd`，HiFormer–TAAC-25 的 B/C/D 三组组件消融，
同样复用对应 H100 seed 20262028 完整 SISA 对照。
截至 2026-09-16，RankMixer–KuaiRand 与 HyFormer–MerRec 的 B/C/D 均已完成并核验，
HiFormer–TAAC-25 尚未形成完整比较。独立论文报告与全部紧凑结果已统一归档于
`sisa_component_report/`，正文为 report.md；该目录保留明确的状态快照。

创建新实验时使用：

```text
experiments/ablations/<study_slug>/
├── README.md          # 从 ../templates/ablation/README.md 复制并填写
├── configs/           # 与主实验不同的参数及配置快照
├── results/           # runs.csv、metrics.csv、summary.csv
├── figures/           # 可从 results 重建的图表
└── notes/             # 调试记录、失败原因和重跑映射
```

建议的 `study_slug` 只使用小写字母、数字和下划线，例如：

- `sisa_score_dim`
- `sisa_lambda_init`
- `sisa_score_scale`
- `sisa_site_selection`
- `seed_robustness`

每个消融必须保持一个明确的控制变量，复用主实验的 baseline，除非报告中说明为什么
需要重跑。正式结论至少要记录模型、数据集、标签、seed、硬件、训练协议、job/task
映射、失败重跑和与控制组的差值。
