# 2026-09-20 HTML汇报更新记录

- 交付页面：experiments/sisa_native_strict/migration/report.html；canonical输入为同目录artifact.json。
- 受众：技术。沿用现有页面、样式和构建器，新增OneTrans诊断区段，不改实验代码。
- 内容结构：完成摘要和指标定义；宏AUC与LogLoss；反向指标说明；最终任务与实验布局；尾批覆盖限制；逐动作精确值。
- 精确查询是本次目的，采用五张原生表格：两张宏指标表、两张逐动作表、一张任务表。已有历史图表完整保留，不新增跨数据集绝对指标图。
- 范围：4个数据集、12次SISA和4次历史baseline；17个动作、68行运行级标签指标。baseline只有seed20262027一次。
- MerRec最终任务636370_9为COMPLETED/0:0，2026-09-20 15:58:35结束；配置、manifest、覆盖计数和双日志最终Test指标核验通过。失败attempt1单独保留。
- 来源：selected_evidence.json、selected_runs.csv、selected_macro_summary.csv、selected_three_seed_summary.csv及两卡覆盖比较表。
- 表格来源中的SQLite查询已实际执行；:runs_json绑定selected_evidence.json的runs，mean/sample_sd聚合器使用statistics.mean/statistics.stdev并忽略NULL。数值与已核验统计逐项一致，六位小数仅为展示格式。
- 原报告所有区段、来源、数据集、表格、指标卡、图表及相对阅读顺序均逐项核对保持不变；新增内容不计入论文accepted 16-cell。
- 重建命令：`node experiments/sisa_native_strict/migration/build_report.mjs`。只生成本地离线页面，不部署站点。
- 页面验收：现有构建器的validation、package、verification全部通过；本地Chromium检查1440px与390px宽度、48个区块、17张表、2张原有图表和4个原有指标卡，来源弹窗及键盘交互通过。源码和HTML内嵌快照一致。

2026-09-20 工作区整理补充：CPU 回归检查发现通用构建器在静态页面中将数字形式的文本
舍入至两位小数，并将长表截为 15 行；上述交互页面验收未覆盖这个问题。现由本仓库构建器
从冻结快照恢复精确展示列及完整静态行数，并补充回归检查。内嵌数据和实验指标保持原值。
