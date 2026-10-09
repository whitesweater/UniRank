# 实验档案

本目录是项目中实验计划、最终报告和可复现结果表的唯一入口。运行时大文件仍保存在
仓库根目录的 `logs/`、`checkpoints/` 和 `artifacts/`；这里只归档适合长期审阅与版本
控制的文档、配置快照、汇总表和图表。

原四模型独立论文实验报告：`sisa_paper_report/report.md`。报告无超链接，正文完整包含实验设置、
结果与全部标签明细，可直接复制给他人阅读。中间材料集中于
`sisa_paper_report/archive/`；原始检查日志和失败截图仅本地归档并忽略。

2026-09-23 起，当前正式范围扩展为 **5 个模型 × 4 个数据集，共 20 个单元**：

- 模型：HiFormer、HyFormer、OneTrans、RankMixer、Zenith。
- 数据集：MerRec（`MerRec_Action`）、QK-Video（`QK_Video_Action`）、
  KuaiRand（`KuaiRand_Video_Action`）、TencentGR（`TencentGR_10M_Action`，旧称 TAAC-25）。

OneTrans 已纳入正式模型范围。现有论文报告及三种子汇总仍保留原 16 个单元的历史统计，
不能直接作为 20 个单元的汇总；更新时须核验 OneTrans 的配对证据并重算指标。
主结果继续采用 seed 20262027 的 baseline/SISA 同协议配对；三轮 SISA 均值是补充描述，
不能称为三轮同协议配对效果。下表中 strict 和 expansion 的规模是历史运行总量，
包含当前范围之外的诊断记录，不作为论文聚合分母。

历史全量可视化：[strict + expansion 报告](sisa_native_strict/migration/report.html)。

## 当前阅读入口

| 用途 | 内容与范围 | 入口 |
| --- | --- | --- |
| 原四模型论文主实验 | 历史 16 cells；seed 20262027 同协议配对；尚未合并 OneTrans | [独立报告](sisa_paper_report/report.md) · [证据归档](sisa_paper_report/archive/) |
| 论文消融正文 | 当前手工修订稿，仅 HyFormer–MerRec | [独立报告](ablations/sisa_component_report/report.md) |
| 组件消融完整归档 | 三组 B/C/D/E；九个新增任务均已完成 | [三组完整报告](ablations/sisa_component_report/archive/reports/three_study_report.md) · [完整结果](ablations/sisa_component_report/archive/results/) |
| HiFormer–QK 诊断 | accepted 单元的现有结果诊断；报告中的优化候选尚未运行 | [诊断报告](hiformer_qk_diagnostic/report.md) · [核验归档](hiformer_qk_diagnostic/archive/) |
| OneTrans 补齐记录 | 已纳入正式模型范围；四数据集三种子已齐；保持协议差异说明 | [研究记录](onetrans_seed_completion/report.md) · [结果口径](onetrans_seed_completion/results/README.md) |

## 研究与历史档案索引

| 实验 | 状态 | 规模 | 入口 |
| --- | --- | ---: | --- |
| SISA 论文实验整理（2026-09-14） | 原四模型主结果已核验；同协议多 seed baseline 尚缺，组件消融已另行完成 | 历史 16 cells / 64 最终运行 / 68 标签组合 | `sisa_paper_report/report.md`（独立完整报告） |
| SISA 原生严格矩阵 | 已完成 | 32 tasks / 16 pairs | [报告](sisa_native_strict/report.md) · [结果](sisa_native_strict/results/) · [迁移证据](sisa_native_strict/migration/) |
| SISA HPC3/ACD 扩展矩阵 | 已完成 | 38 tasks / 19 pairs | [报告](sisa_expansion_acd/report.md) · [结果](sisa_expansion_acd/results/) · [原计划](sisa_expansion_acd/planning.md) |
| SISA seed 20262028 | 已完成 | 16 tasks / 68 labels | [结果](sisa_single_seed20262028/results/) · [与 seed 20262027 比较](sisa_single_seed20262028/comparison/) |
| SISA seed 20262029 | 已完成 | 16 tasks / 68 labels | [协议](sisa_single_seed20262029/protocol.md) · [结果](sisa_single_seed20262029/results/) · [与 seed 20262028 比较](sisa_single_seed20262029/comparison_vs_seed20262028/) |
| SISA 三-seed 统一分析 | 已完成 | 3 seeds / 16 cells / 68 labels | [摘要](sisa_three_seed_unified/results/summary.md) · [HTML 报告](sisa_three_seed_unified/report/report.html) · [来源](sisa_three_seed_unified/sources/) |
| OneTrans SISA 三-seed 补齐 | 4数据集三种子已齐；MerRec重提636370_9成功 | 8个新增逻辑任务全部完成，含1次成功重提；Taobao已取消 | [HTML总报告](sisa_native_strict/migration/report.html) · [明细](onetrans_seed_completion/report.md) · [选定结果](onetrans_seed_completion/results/selected_macro_summary.csv)（旧报告中的诊断分类为历史口径） |
| OneTrans 两卡备选分支 | 选择结束；KuaiRand/TAAC-25 采用两卡，其余采用四卡 | 八对任务的选择及取消回执 | [协议](onetrans_seed_completion_ws2/README.md) · [选择快照](onetrans_seed_completion_ws2/notes/race_completion.json) |
| SISA 组件消融 | 三组均已完成；当前论文正文仅展示 HyFormer–MerRec | 9 个新增 B/C/D + 3 个历史 E | [消融索引与规范](ablations/README.md) · [报告说明](ablations/sisa_component_report/README.md) |
| HiFormer–QK 诊断 | 现有证据已归档 | 1 个配对单元 + 2 次描述性 SISA 运行 | [报告](hiformer_qk_diagnostic/report.md) · [来源说明](hiformer_qk_diagnostic/archive/source_notes.md) |

## 统一目录约定

### 近期研究入口

| 研究 | 正式阅读入口 | 复现与证据 |
| --- | --- | --- |
| RISE 样本、机制与补充实验 | `rise_sample_analysis/final_research_report.md` | 同目录 `scripts/`、`tests/`、`results/`、`figures/` 和 `paper/` |
| RISE 注意力案例 | `rise_attention_case_evidence/report.md` | 同目录协议、行为分组、结果及审计回执 |
| RISE 注意力重分配 | `rise_attention_redistribution/report.md` | 同目录协议、结果、图表及审计回执 |
| RISE 语义案例 | `rise_attention_semantic/report.md` | 同目录元数据审计、筛选记录、结果及图表 |
| RISE 展示案例 | `rise_attention_showcases/report.md` | 同目录分组、结果及图表 |
| HiFormer target-only 四数据集扩展 | `hiformer_target_only_extension/report.md` | 同目录计划、最终结果、提交及完成回执 |
| HiFormer target-only 多种子 | `hiformer_target_only_multiseed/report.md` | 同目录计划、最终结果、参考审计及完成回执 |
| SISA 种子波动与复跑 | `sisa_seed_variability_20260923/final_result_report.md` | 同目录 `results/`、`audit/`、`rerun/` |
| SISA 当前结果选择器 | `sisa_three_seed_unified/report/report.html` | 同目录 README、构建脚本、来源目录及选择器核验 |
| HiFormer–QK 失败分析 | `hiformer_qk_diagnostic/failure_analysis_20260924/report.md` | 同目录生成器、证据摘要及核验回执 |

### RISE 补充诊断与审计入口

下列目录保存固定 checkpoint 和诊断窗口上的干预、重复与统计核验，不代表新增的
20-cell 主实验汇总。索引仅整理已有材料；各目录计划和审计文件定义其证据范围。

| 研究 | 协议与代码入口 | 紧凑结果与核验 |
| --- | --- | --- |
| RankMixer–KuaiRand 固定窗口评分干预 | [协议](rise_discussion_validation_20260925/README.md) | [汇总](rise_discussion_validation_20260925/results_651654/analysis/summary.json) · [独立审计](rise_discussion_validation_20260925/portable_records/independent_audit.json) |
| HiFormer–TencentGR 独立诊断 | [协议](rise_discussion_validation_hiformer_20260925/README.md) | [汇总](rise_discussion_validation_hiformer_20260925/results_651691/analysis/summary.json) · [独立审计](rise_discussion_validation_hiformer_20260925/portable_records/independent_audit.json) |
| 历史顺序置换及独立分析 | [执行协议](rise_history_permutation_20260925/README.md) · [分析协议](rise_history_permutation_analysis_20260925/README.md) | [紧凑结果审计](rise_history_permutation_analysis_20260925/completed_results_audit/compact_result_summary.json)；旧版本协议与分析器保留在各自目录 |
| 评分干预 seed 20262029 重复与结果核验 | [重复脚本](rise_score_profile_repeats_20260926/run_repeat.py) · [核验脚本](rise_score_profile_results_audit_20260926/compare_profiles.py) | [HiFormer 比较](rise_score_profile_results_audit_20260926/hiformer_checked/comparison.md) · [RankMixer 比较](rise_score_profile_results_audit_20260926/rankmixer_checked/comparison.md) |
| 固定窗口 AUC 区间与多重比较 | [计划阅读副本](rise_auc_significance_20260926/portable_records/plan.json) · [计算脚本](rise_auc_significance_20260926/run_auc.py) | [区间结果](rise_auc_significance_20260926/results_652845/auc_intervals.md) · [独立复核](rise_auc_significance_20260926/results_652845/review.md) |
| 已取消的 frozen-backbone adaptation | [取消说明](rise_frozen_adaptation_20260925/CANCELLED.md) | 未提交、未运行；准备材料仅留本地，不作为实验结果 |

上述诊断脚本从文件位置解析仓库根目录，Slurm 启动器从仓库根目录提交。原始冻结计划、
审计和逐运行源码副本保持原样并仅留本地；`portable_records/` 提供替换本机路径后的
阅读副本及原件/副本散列。阅读副本不能替代原件进行历史散列校验，也不能作为执行计划。
脚本的路径调整改变了源码散列；重新执行需在新 study 中重新冻结计划，历史复核使用原始
冻结源码与运行产物。各 study 的 `portable_records/README.md` 说明归档范围。

### 文件保留规则

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

正式报告、最终 CSV/JSON、协议、复现脚本、测试、论文源文件与正式图表应提交 Git。
`notes/submission.json`、`notes/reference_audit.json` 等可能是采集器输入，不能因为位于
`notes/` 或 `audit/` 就整目录忽略。记录口径变化、候选筛选或结论修订的历史证据也应保留。

以下产物保留在本地，由根目录 `.gitignore` 排除：

- `results/snapshots/`、`results/*_snapshots/`：周期采集快照；相邻的最终结果表仍提交。
- `notes/*.log`、`audit/*.log`、原始调度器转储及明确列出的监控进程状态：本次运行的检查材料。
- 三种子报告中合并前和排版前的重复备份：最终选择器之前的完整报告仍作为历史档案提交。
- `rise_sample_analysis/RISE_final_delivery.zip`：由已有研究材料打包的本地交付副本。
- 仓库根目录 `document.pdf`：用户提供的原始稿件输入，跨机器复核时需单独提供；正式补充材料
  和科研图表 PDF 仍提交。各级 `.venv/` 仅保存本机环境，依赖元数据和锁文件应提交。
- 实验目录中的 `cache-*/`、`runtime_cache/`、`torchinductor/`、`triton/` 及 `*.out`、
  `*.log`：编译缓存和完整运行日志；`benchmark/` 的上游参考日志继续保留。
- 实验目录中的 NPZ/NPY、PT/PTH、Parquet、`frozen_meta.csv`，以及
  `rise_discussion_validation*` 和 `rise_score_profile_repeats_*` 中的 `meta.csv`、
  `predictions.csv`、`results*` 下的 `row_diagnostics.csv`、`cluster_loss_sums.csv`：
  原始数组、逐样本数据和可再生成的逐用户统计。最终区间、指标、对比、汇总及用户数量表，
  以及 `rise_attention_*` 中已经筛选的紧凑案例元数据和预测表继续提交。
  重新计算统计量或逐行核验时仍需这些本地原始产物，单独克隆仓库不足以完成此类复核。
- `results*/progress.json` 和 `synthetic_e2e*/`：运行进度及生成的合成测试现场；生成它们的
  测试源码继续保留。完成回执、数值门槛核验和最终统计继续提交。
- RISE 固定窗口诊断和 seed 重复中的 `results*/plan.json`、`frozen_sources/`、完整
  `manifest.json`、逐模板 `template_receipts.json` 和重复的 `training_params.json`：
  同内容副本及详细运行现场记录，原件保持不变并仅留本地。历史置换分析的传输清单和重复
  启动器也仅留本地；有行为差异的旧分析器、测试与版本说明继续提交。
- 上述新增 RISE 研究中含本机绝对路径的原始计划与审计：Git 保存 `portable_records/`
  中的路径可移植阅读副本，原件用于历史散列校验。原始完成清单仍可能引用本地副本和大文件。
- `rise_frozen_adaptation_20260925/` 的所有准备材料：实验在提交前已取消，仅保留
  `CANCELLED.md`；代码、启动器、设计和冻结计划继续留在本地。

冻结的 `delivery_manifest.json`、`final_delivery.json` 等是当时交付与审计材料的清单，
可能包含上述本地快照或日志，不等同于 Git 文件清单；保留原路径及散列，不因整理而重写
历史记录。仅克隆仓库即可阅读正式成果，完整复核运行现场仍需相应的本地证据。

新增消融实验时，从 [消融模板](templates/ablation/README.md) 开始，并同步更新本页索引。

## 完成一项研究后的整理

1. 确定唯一的最终阅读入口；手工修订正文与可再生成的完整归档使用不同输出路径。
2. 将紧凑结果、冻结配置、提交与取消回执放回所属 study；采集与提交工具放在 `scripts/`，
   相应回归测试放在 `tests/`。原始日志、模型和临时检查输出留在 Git 忽略的运行目录。
3. 同步本页及 study README 的范围、完成状态和重建命令；旧快照须注明口径，避免与最终选定结果混读。
4. 检查差异并完成必要验证，按研究主题提交；用 `git status --short` 确认没有遗漏的成果。
   整理前快照与检查日志可保存在 `artifacts/workspace_cleanup/`。
