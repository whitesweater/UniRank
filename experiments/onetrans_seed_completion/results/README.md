# OneTrans 结果文件口径

现有研究记录为上一级 `report.md`，采用 2026-09-20 核验的 `selected_` 文件。
2026-09-23 起，OneTrans 已纳入正式 20-cell 范围；这些文件仍保留原核验快照，
尚未合并到原四模型 16-cell 报告。后续合并须核验配对证据并重新计算聚合指标。

| 文件 | 用途 |
| --- | --- |
| `selected_evidence.json` | 最终选定运行、核验问题、来源散列、调度器快照与失败 attempt |
| `selected_runs.csv` | 16 次选定运行：4 次历史 baseline 与 12 次 SISA；含布局、job/task 与宏指标 |
| `selected_label_metrics.csv` | 68 行运行级标签指标 |
| `selected_macro_summary.csv` | 每数据集的宏指标及三种子描述性统计 |
| `selected_three_seed_summary.csv` | 每数据集、每标签的三种子描述性统计 |

QK/MerRec 保留四卡分支；KuaiRand/TAAC-25 新增种子采用两卡分支。分支选择快照位于
`experiments/onetrans_seed_completion_ws2/notes/race_completion.json`；MerRec 最终重提记录位于
`../notes/retry_completion.json`。布局与覆盖差异在报告中单独说明。

无 `selected_` 前缀的文件保留四卡专用采集器的历史快照，包含原始任务编号及取消状态。
它们不包含两卡选定结果，不能据此判断当前仍有待补种子，也不能与 `selected_` 表叠加计数。
`scripts/collect_onetrans_seed_completion.py` 只更新这些四卡文件。

`selected_` 文件是已审核的冻结快照；目前未归档统一重建这些混合布局结果的采集脚本。
重新核验时需按 evidence 中的来源、各自布局及明确的 attempt 选择重做审计，不能只运行
四卡采集器就覆盖本报告。普通文件整理保留快照及其来源散列。
