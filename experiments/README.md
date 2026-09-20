# 实验档案

本目录是项目中实验计划、最终报告和可复现结果表的唯一入口。运行时大文件仍保存在
仓库根目录的 `logs/`、`checkpoints/` 和 `artifacts/`；这里只归档适合长期审阅与版本
控制的文档、配置快照、汇总表和图表。

独立论文实验报告：`sisa_paper_report/report.md`。报告无超链接，正文完整包含实验设置、
结果与全部标签明细，可直接复制给他人阅读。中间材料集中于
`sisa_paper_report/archive/`；原始检查日志和失败截图仅本地归档并忽略。

当前论文证据只采用 HiFormer、HyFormer、RankMixer、Zenith × QK-Video、KuaiRand、
TAAC-25、MerRec 的 16 个单元。主结果采用 seed 20262027 的 baseline/SISA 同协议
配对；三轮 SISA 均值是补充描述，不能称为三轮同协议配对效果。下表中 strict 和
expansion 的规模是历史运行总量，包含当前范围之外的诊断记录，不作为论文聚合分母。

历史全量可视化：[strict + expansion 报告](sisa_native_strict/migration/report.html)。

## 当前阅读入口

| 用途 | 内容与范围 | 入口 |
| --- | --- | --- |
| 论文主实验 | accepted 16 cells；seed 20262027 同协议配对 | [独立报告](sisa_paper_report/report.md) · [证据归档](sisa_paper_report/archive/) |
| 论文消融正文 | 当前手工修订稿，仅 HyFormer–MerRec | [独立报告](ablations/sisa_component_report/report.md) |
| 组件消融完整归档 | 三组 B/C/D/E；九个新增任务均已完成 | [三组完整报告](ablations/sisa_component_report/archive/reports/three_study_report.md) · [完整结果](ablations/sisa_component_report/archive/results/) |
| HiFormer–QK 诊断 | accepted 单元的现有结果诊断；报告中的优化候选尚未运行 | [诊断报告](hiformer_qk_diagnostic/report.md) · [核验归档](hiformer_qk_diagnostic/archive/) |
| OneTrans 补齐诊断 | 独立于 accepted 16-cell；四数据集三种子已齐 | [诊断报告](onetrans_seed_completion/report.md) · [结果口径](onetrans_seed_completion/results/README.md) |

## 研究与历史档案索引

| 实验 | 状态 | 规模 | 入口 |
| --- | --- | ---: | --- |
| SISA 论文实验整理（2026-09-14） | 主结果已核验；同协议多 seed baseline 尚缺，组件消融已另行完成 | accepted 16 cells / 64 最终运行 / 68 标签组合 | `sisa_paper_report/report.md`（独立完整报告） |
| SISA 原生严格矩阵 | 已完成 | 32 tasks / 16 pairs | [报告](sisa_native_strict/report.md) · [结果](sisa_native_strict/results/) · [迁移证据](sisa_native_strict/migration/) |
| SISA HPC3/ACD 扩展矩阵 | 已完成 | 38 tasks / 19 pairs | [报告](sisa_expansion_acd/report.md) · [结果](sisa_expansion_acd/results/) · [原计划](sisa_expansion_acd/planning.md) |
| SISA seed 20262028 | 已完成 | 16 tasks / 68 labels | [结果](sisa_single_seed20262028/results/) · [与 seed 20262027 比较](sisa_single_seed20262028/comparison/) |
| SISA seed 20262029 | 已完成 | 16 tasks / 68 labels | [协议](sisa_single_seed20262029/protocol.md) · [结果](sisa_single_seed20262029/results/) · [与 seed 20262028 比较](sisa_single_seed20262029/comparison_vs_seed20262028/) |
| SISA 三-seed 统一分析 | 已完成 | 3 seeds / 16 cells / 68 labels | [摘要](sisa_three_seed_unified/results/summary.md) · [HTML 报告](sisa_three_seed_unified/report/report.html) · [来源](sisa_three_seed_unified/sources/) |
| OneTrans SISA 三-seed 补齐 | 4数据集三种子已齐；MerRec重提636370_9成功 | 8个新增逻辑任务全部完成，含1次成功重提；Taobao已取消 | [HTML总报告](sisa_native_strict/migration/report.html) · [明细](onetrans_seed_completion/report.md) · [选定结果](onetrans_seed_completion/results/selected_macro_summary.csv)（独立诊断，不混入accepted 16-cell） |
| OneTrans 两卡备选分支 | 选择结束；KuaiRand/TAAC-25 采用两卡，其余采用四卡 | 八对任务的选择及取消回执 | [协议](onetrans_seed_completion_ws2/README.md) · [选择快照](onetrans_seed_completion_ws2/notes/race_completion.json) |
| SISA 组件消融 | 三组均已完成；当前论文正文仅展示 HyFormer–MerRec | 9 个新增 B/C/D + 3 个历史 E | [消融索引与规范](ablations/README.md) · [报告说明](ablations/sisa_component_report/README.md) |
| HiFormer–QK 诊断 | 现有证据已归档 | 1 个配对单元 + 2 次描述性 SISA 运行 | [报告](hiformer_qk_diagnostic/report.md) · [来源说明](hiformer_qk_diagnostic/archive/source_notes.md) |

## 统一目录约定

每个正式实验使用一个稳定的 `study_slug`：

```text
experiments/<study_slug>/
├── report.md          # 结论、协议、异常恢复、局限性
├── planning.md        # 实验矩阵与完成清单（如适用）
├── configs/           # 本次实验专用配置或配置快照（如适用）
├── results/           # runs/metrics/paired summary 等小型机器可读结果
├── figures/           # 从 results 生成的图表
├── migration/         # 外部节点迁入的原始报告和审计证据（如适用）
├── archive/           # 冻结证据、历史完整报告和该报告专用的生成脚本（如适用）
└── notes/             # 非最终分析记录；不得替代 report.md
```

目录中不放 Parquet、checkpoint、完整 Slurm stdout 或编译缓存。`report.md` 必须记录原始
日志位置、最终 job/task 映射和结果生成命令，使运行时证据仍可追溯。

新增消融实验时，从 [消融模板](templates/ablation/README.md) 开始，并同步更新本页索引。

## 完成一项研究后的整理

1. 确定唯一的最终阅读入口；手工修订正文与可再生成的完整归档使用不同输出路径。
2. 将紧凑结果、冻结配置、提交与取消回执放回所属 study；采集与提交工具放在 `scripts/`，
   相应回归测试放在 `tests/`。原始日志、模型和临时检查输出留在 Git 忽略的运行目录。
3. 同步本页及 study README 的范围、完成状态和重建命令；旧快照须注明口径，避免与最终选定结果混读。
4. 检查差异并完成必要验证，按研究主题提交；用 `git status --short` 确认没有遗漏的成果。
   整理前快照与检查日志可保存在 `artifacts/workspace_cleanup/`。
