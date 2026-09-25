# 自主论文增强：执行要求与完成证据

用户已授权自主完成分析、补充必要实验及详细学术文档。另一个 agent 负责的
`sisa_score_dim` 脚本、冻结配置、任务 643736 和输出属于独立工作，不修改、不取消、不重复提交。

本研究的成功标准是实测分析与完整论文材料，不以仅提交作业、实现脚本或写出计划替代完成。

| 工作包 | 必须出现的最终证据 | 当前进度入口 |
|---|---|---|
| 原始结果与论文对齐 | 36份日志/312指标复核、16格68标签完整表 | results/audit_summary.json；已完成 |
| 同样本组件分组 | B/C/D/E全部测试预测、11599872行身份一致、全局metric gate、逐动作组表与不确定性 | scripts/export_predictions.py；GPU验证待完成 |
| 数据语义与分组旁表 | 训练目标频次、窗口与类别定义、全量支持度、未知/歧义率、时序归因边界 | scripts/build_group_features.py；CPU验证进行中 |
| 原生Baseline匹配 | 2028 HyFormer–MerRec、HiFormer–QK/TAAC三个新baseline训练完成、协议/覆盖/日志审核 | notes/matched_baseline_submission.json；643809 |
| 完整方法样本效果 | 原生Baseline与既有RISE在同一2028评估样本上分组；不冒充2027原实例 | 依赖新baseline完成 |
| 累积/固定控制 | RankMixer–KuaiRand两个新control的完成记录与E/B/C/D明确对照；参数与协议审核 | scripts/run_rankmixer_controls.py |
| 内部机制观测 | 固定前向表示下修正范围、TV、幅度/相位与padding的实测统计；输出一致性 | 独立诊断脚本待实现 |
| 计算成本 | 同硬件、同batch与编译状态的暖机后重复延迟/显存测量，清楚区分纯前向与数据加载 | 独立benchmark待实现 |
| 学术交付 | 更新的独立中文报告、可插入论文的英文LaTeX段落/表格/图、完整附录和实验审计记录 | research_report.md后续更新 |
| 最终验收 | 所有正式行数/协议/数值检验、CPU回归与新测试、图表视觉检查、失败尝试留档 | notes/validation.md后续更新 |

新增baseline严格读取历史2028完整RISE日志配置，仅关闭SISA并改变输出身份与目录；
这三项不会恢复2027的原模型。所有新训练都使用两张H100、每卡16384、1epoch、
原loader与辅助初始化子种子。训练输出分别隔离，不覆盖旧模型。

统计区间计划增加一种可扩展的用户簇影响函数方法，明确它是固定已训练模型下的
渐近cluster-sandwich区间，不是精确bootstrap，也不覆盖训练seed方差。
算法需通过ties、有限差分、同模型配对方差为零和跨组协方差检查，并与小规模
真实用户bootstrap进行敏感性比较，之后才用于正式结论。

阶段记录：此前完成的是可行性审计与研究方案，构成具体进展；本阶段继续实现并运行。
调度等待必须以实际squeue/sacct作业句柄为准，不因暂时没有输出重启作业。
