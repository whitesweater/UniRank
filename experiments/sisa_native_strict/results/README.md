# SISA 原生严格矩阵机器可读结果

本目录保存 32-task strict 实验适合长期审阅和版本控制的小型结果表：

- [`runs.csv`](runs.csv)：32 个训练任务的 Slurm 映射、协议检查、完成标记和证据路径；
- [`metrics.csv`](metrics.csv)：152 行原始指标，即 76 个标签的 baseline/SISA 结果；
- [`summary.md`](summary.md)：76 个标签级配对的 AUC、Logloss 和差值表。

这些文件于 2026-08-25 按
[`migration/migration_files.txt`](../migration/migration_files.txt)
从历史实验节点迁移到当前归档节点，并与远端执行了 rsync checksum 复核。迁移后审计
结果为：任务 `32/32` 完成且协议有效，76 个 AUC 配对中
50 个提升、26 个下降，平均差值 `+0.002115`；16 个模型/数据集单元中 12 个为正。

完整迁移包还包括 32 份 Slurm 日志、32 份训练指标日志和
[离线可视化报告](../migration/report.html)。运行时原始
副本继续保存在仓库根目录的 `artifacts/sisa_native_strict/`、`logs/` 和
`checkpoints/`，本目录中的 CSV/Markdown 是统一实验档案副本。

如需从原始日志重新生成，可运行 `scripts/collect_sisa_native_strict_results.py`，并按
[strict 报告](../report.md)中的 32-task override 映射核验。
