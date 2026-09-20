# DIGAI Lab strict 实验迁移证据

本目录的旧混合矩阵是历史诊断档案，不用于当前论文汇总。Accepted 16-cell 与
Diagnostic 逐动作视图分别展示；当前论文以 `experiments/sisa_paper_report/report.md` 为准。

2026-09-20更新：同一HTML页面新增独立的OneTrans三种子诊断区段，包含四个数据集的
宏AUC/LogLoss、17个动作标签明细和八个新增任务的最终状态。MerRec重提636370_9已成功完成。
原有区段、图表与数据保持不变；OneTrans不计入accepted 16-cell统计。新增来源为
`experiments/onetrans_seed_completion/results/selected_`前缀文件。

页面生成器会保留静态表格中预格式化指标的精度、差值符号和完整行数，避免离线或打印时
被通用构建器舍入为两位小数。此处理使用同一 artifact.json，不修改原始指标或内嵌证据。

本目录集中保存 2026-08-25 从 DIGAI Lab 迁入的原始 strict 报告和离线审计包。正式结论请从
[统一报告](../report.md)和[机器可读结果](../results/)进入；本目录用于追溯迁移来源，
不作为新的并列报告根目录。

| 文件 | 用途 |
|---|---|
| [`source_report.md`](source_report.md) | DIGAI Lab 原始完整 Markdown 报告 |
| [`report.html`](report.html) | 自包含的离线可视化总报告，含 strict 与 38-task expansion 新结果 |
| [`migration_files.txt`](migration_files.txt) | DIGAI Lab 源目录下的 75 文件精确迁移清单 |
| [`artifact.json`](artifact.json) | 生成可视化总报告时使用的 canonical artifact |
| [`build_report.mjs`](build_report.mjs) | 使用 Data Analytics 可移植阅读器重新生成并校验美化后的自包含 HTML |
| [`audit_snapshot.csv`](audit_snapshot.csv) | 迁移清单审计快照 |
| `combined_headline.sql`、`expansion_unit_summary.sql`、`combined_auc_matrix.sql`、`unit_deltas.sql`、`migration_inventory.sql` | 总报告统计查询；矩阵查询合并 DIGAI Lab 的 16 个 strict 单元与 HPC3 的 19 个 expansion 单元；`headline.sql` 保留旧 strict 快照 |

`migration_files.txt` 保留 DIGAI Lab 迁移当时的原始相对路径，仅用于审计追溯；
`artifact.json` 已更新为当前仓库路径，并合并 38-task expansion 结果。32 份 strict Slurm
日志和训练指标日志仍分别保存在根目录的 `logs/` 与 `checkpoints/`，适合版本控制的最终
结果统一归档在 `experiments/`。

重新生成报告（更新本地 `artifact.json` 和 `report.html`，不部署、不启动服务）：

```bash
node experiments/sisa_native_strict/migration/build_report.mjs
```

脚本会先从现有 strict/expansion 结果只读生成报告中的逐动作视图，再调用本机已安装的
Data Analytics 报告构建器打包离线 HTML。它只更新报告文件，不修改实验结果 CSV；
Python 环境仍统一由项目根目录的 `uv` 配置管理。

逐动作来源查询保存在 `action_breakdown.sql`：构建脚本将已核对的 CSV 配对行绑定为
SQLite JSON 参数，执行只读展示查询。生成结果保留六位小数，并明确区分论文范围与诊断记录。

### 图表映射

| 报告区段 | 图形与字段 | 用途与配色 |
|---|---|---|
| 70-task 完整矩阵 | 发散热力图；行=`dataset`，列=7 个模型，值=平均 ΔAUC ×10⁻³ | 35/35 单元完整覆盖；绿=提升、红=下降，颜色深浅按全矩阵最大绝对值对称映射；蓝色内框=DIGAI Lab 迁移记录；行名和列名旁的 `−` 可折叠，顶部标签可逐项恢复或一键恢复全部 |
| 32-task strict | 横向条形图；类别=`unit`，值=`delta_auc` | 回顾 16 个 strict 单元的排序与正负方向 |

迁移验证结果：75/75 文件存在，合计 253,466,283 bytes；rsync checksum 复核无内容
差异。正式 strict 模型权重在旧训练流程中已被删除，因此迁移包不含 `.model` 文件。
