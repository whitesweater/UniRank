# 自主论文增强：最终执行要求与完成证据

本研究最终入口为 final_research_report.md；初始执行清单原样保留在 archive/execution_requirements_initial.md。

| 工作包 | 完成证据与范围 |
|---|---|
| 原稿与主表审计 | 36日志、312指标、正式16格68标签；results/audit_summary.json |
| 14项新增训练 | new_training_runs.csv五项、targeted_training_runs.csv三项、site_replication_runs.csv六项；全部eligible且COMPLETED/0:0 |
| 同样本组件及原生分析 | MerRec 11599872行、QK 48529408行、TAAC 153157632行；正式身份/全局metric gate通过 |
| 数据语义与人口统计 | 全训练频次、窗口类别/歧义处理、人口构成完成；历史动作时点可得性未解决，真实影响比例未知 |
| 统计不确定性 | 用户簇配对影响函数、同用户跨组协方差、精确AUC组对分解、小规模真实用户bootstrap验证 |
| 机制控制 | RankMixer局部/固定、HyFormer固定、HiFormer/Zenith/HyFormer位置控制全部完成；明确容量差别与单seed范围 |
| 输出不变的机制观测 | 编译路径B/E共6144条site观测通过；E真实AM/PA内部量未通过一致性，明确拒绝使用，不声称全部状态采集成功 |
| 计算成本 | 同H100、A/B/E、三种batch、正反顺序共6轮18进程；延迟、内存及实际attention后端 |
| 预测质量 | ROC-AUC之外的AP、Logloss、Brier、20共同概率bin；负向标签完整保留 |
| 学术交付 | 独立完整中文报告、可插入英文LaTeX、科学图、compact CSV、审计回执、失败记录与复现代码 |
| 最终核查 | notes/final_delivery_validation.md；原始合并进程SIGSEGV与模块隔离通过回执均保留 |

没有修改核心训练代码、根训练依赖或另一项score_dim消融。初始8项与追加6项均从冻结配置运行，保留历史20262027主表；不将后选配置回填主表。区间只描述固定训练模型的样本不确定性，不能证明训练seed稳定。新任务与历史E的日志协议/覆盖对齐，但历史软件与数据字节身份未全部追溯。
