# 消融实验索引

后续所有 SISA 或 UniRank 消融实验都放在本目录下，每个研究问题一个子目录，不按临时
Slurm job ID 建目录。

当前尚无正式完成的消融 study。创建新实验时使用：

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

