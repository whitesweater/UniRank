# 消融实验索引

后续所有 SISA 或 UniRank 消融实验都放在本目录下，每个研究问题一个子目录，不按临时
Slurm job ID 建目录。

当前 study：`sisa_components_bcd`，RankMixer–KuaiRand 的 B/C/D 三组组件消融，
复用 H100 完整 SISA 对照。提交状态见该目录的 notes/launch.md。
追加 study：`sisa_components_hyformer_merrec_bcd`，HyFormer–MerRec 的 B/C/D 三组组件消融，
复用 H100 seed 20262028 完整 SISA 对照；具体记录位于该目录。
第三组：`sisa_components_hiformer_taac_bcd`，HiFormer–TAAC-25 的 B/C/D 三组组件消融，
同样复用对应 H100 seed 20262028 完整 SISA 对照。
截至 2026-09-17 14:19:22（北京时间），三组 B/C/D 共九个新增任务均已完成并核验，
连同三个历史 E 对照，共形成三组完整 B/C/D/E 比较。材料统一归档于
`sisa_component_report/`：当前 `report.md` 为仅展示 HyFormer–MerRec 的手工修订正文；
三组完整报告位于 `archive/reports/three_study_report.md`，全部紧凑结果仍在
`archive/results/`。正文展示范围缩小不代表其他两组结果丢失或尚未完成。

维度研究：`sisa_score_dim`，仅 HyFormer–MerRec。比较每头 8、16、32、64 维，
16 维展示主实验三个种子的均值；548166_1 仅保留为原启动协议参考。8、32、64 维各为一次新增训练。
截至 2026-09-23 11:13，四种维度均已完成审计。64 维最终用四卡、每卡 batch 8192，
全局 batch 保持 32768，DDP 与 drop_last 覆盖差异已披露。
独立完整报告为该研究的 `report.md`，提交与守护记录位于 `notes/`，紧凑结果位于 `results/`。

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
