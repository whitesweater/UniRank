# SISA 组件消融论文材料

独立中文正文为 report.md，包含实验动机、B/C/D/E 设计、固定训练设置、数据与评估口径、
两组完整正式结果、逐标签表、第三组状态及可复制的论文表述。正文没有超链接。

本目录按单次审阅快照归档，不随运行日志自动更新。当前快照有两组完整比较，
HiFormer–TAAC-25 的 B/C/D 仍未形成完整比较。下一次刷新必须重新核验并审阅结论。

## 文件组织

- report.md：给论文作者及合作者阅读的独立报告。
- archive/results/evidence.json：严格核验后的指标、配置检查、覆盖与原始证据散列。
- archive/results/runs.csv：运行状态与宏指标；未通过核验的任务指标留空。
- archive/results/label_metrics.csv：全部合格运行的逐标签指标。
- archive/results/comparisons.csv：仅完整 B/C/D/E 组合的逐标签及宏平均差值。
- archive/notes/independent_audit.md：独立解析核验、反例与证据边界。
- archive/notes/source_notes.md：来源、交付形式及验证说明。
- archive/scripts/build_report.py：由冻结 evidence.json 重建当前报告。

原始日志、运行 manifest、outcome 和模型分别留在 logs/、artifacts/、checkpoints/，
这些运行目录由 Git 忽略，不复制到报告附件。正式采集程序位于
scripts/collect_sisa_component_results.py；它读取原始结果，不提交训练任务。

## 重建与更新

在仓库根目录执行：

```bash
.venv/bin/python scripts/collect_sisa_component_results.py
.venv/bin/python experiments/ablations/sisa_component_report/archive/scripts/build_report.py
```

采集器会刷新状态与通过审计的结果；生成器针对本次已审阅的两组解释编写，
如果第三组全部完成，它会停止并要求先更新结果解读，避免新表格与旧结论混用。
未通过核验的记录保留具体问题，不使用验证集或未完成作业的最终段充当正式结果。
