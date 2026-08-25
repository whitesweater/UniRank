# HPC01 strict 实验迁移证据

本目录集中保存 2026-08-25 从 HPC01 迁入的原始 strict 报告和离线审计包。正式结论请从
[统一报告](../report.md)和[机器可读结果](../results/)进入；本目录用于追溯迁移来源，
不作为新的并列报告根目录。

| 文件 | 用途 |
|---|---|
| [`source_report.md`](source_report.md) | HPC01 原始完整 Markdown 报告 |
| [`report.html`](report.html) | 自包含的离线可视化迁移报告 |
| [`migration_files.txt`](migration_files.txt) | HPC01 源目录下的 75 文件精确迁移清单 |
| [`artifact.json`](artifact.json) | 生成可视化报告时使用的结构化数据 |
| [`audit_snapshot.csv`](audit_snapshot.csv) | 迁移清单审计快照 |
| `headline.sql`、`unit_deltas.sql`、`migration_inventory.sql` | 报告统计查询 |

`migration_files.txt` 和 `artifact.json` 中的 `reports/...`、`SISA_STRICT_EXPERIMENTS.md`
路径是 HPC01 源项目在迁移当时的相对路径，为保持审计证据原样而没有重写。迁入 HPC3
后，这些小型文件统一收纳在本目录；32 份 Slurm 日志、32 份训练指标日志和运行时 CSV
仍分别保存在根目录的 `logs/`、`checkpoints/` 和 `artifacts/sisa_native_strict/`。

迁移验证结果：75/75 文件存在，合计 253,466,283 bytes；rsync checksum 复核无内容
差异。正式 strict 模型权重在旧训练流程中已被删除，因此迁移包不含 `.model` 文件。
