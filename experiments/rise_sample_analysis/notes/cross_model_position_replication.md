# 跨模型接入位置复核：用户追加要求与冻结方案

2026年9月23日，用户要求把HiFormer的接入位置发现扩展到其他模型验证。
本轮先固定QK-Video及seed20262028协议，比较架构差异，不同时扩展数据集或调整训练预算。
所有结果，包括与HiFormer不一致的结果，都必须进入最终报告。

## 为什么需要复核，而不能直接推广

已完成的HiFormer–QK对照：原生AUC0.93343800；完整RISE0.93089125；仅历史汇聚
0.93441475；仅字段层0.92939075。这支持本例存在位置依赖，但还不能说明所有模型的
字段/token attention都不适合RISE。

Zenith有历史target pooling与聚合token attention，适合直接复核同类位置差异。
其token中也包含历史汇聚信息，因此“非事件序列token”不等于“完全没有历史信息”。
HyFormer的既有RISE位置是历史self-attention和query-to-sequence decoding；两者都
属于序列处理，本轮检验的是阶段差异，不把query decoding重命名为字段attention。
RankMixer只在一个原生target pooling位置接入RISE；没有第二类既有位置可做同样拆分，
因此作为结构参照保留，不新增注意力模块来制造一个不等价的消融。

## 六项训练及其比较关系

Slurm数组：644632，0–5，最多并行2项。每项两张H100、16 CPU、480GiB主机内存。

| task | 模型 | 设置 | 保留RISE位置数 | 可训练稠密参数 |
|---:|---|---|---:|---:|
| 0 | Zenith | 原生Baseline | 0 | 11,164,164 |
| 1 | HyFormer | 原生Baseline | 0 | 7,505,412 |
| 2 | Zenith | history_only | 1个target pooling | 11,167,446 |
| 3 | HyFormer | sequence_only | 3个历史self-attention | 7,568,400 |
| 4 | Zenith | token_only | 3个聚合token attention | 11,227,152 |
| 5 | HyFormer | query_only | 3个query decoder | 7,568,400 |

两个完整RISE参照复用各自已存的seed20262028检查点。每个模型最终同时报告原生、
完整RISE、两种单位置配置的四动作AUC、Logloss和宏平均。位置配置改变活跃参数量，
不把这一研究称为等参数控制；HyFormer两种单位置配置稠密参数量恰好相同，仍应记录
其计算角色不同。单种子结果也不能提供训练方差或跨seed显著性。

## 冻结协议和启动前证据

- 所有有效Params直接由对应历史E日志恢复。Baseline仅改变model_id和sisa_enabled；
  单位置配置仅改变model_id，处理条件由独立进程内的构建patch记录。
- seed20262028、loader2027、aux-init20260822、每卡16384、global32768、1epoch、
  max_len100、BF16和torch.compile保持一致；核心代码、根配置和另一项score_dim研究不改。
- 当前QK训练/验证/测试逐rank步数分别为12038/12036、1527/1527、1481/1481，与两个
  历史参照相同；相应全局保留行数394,428,416、50,036,736、48,529,408。
- 4项CPU测试使用真实模型结构和缩小词表，验证全部保留参数的初始值、全局RNG状态、
  接入点数量、reference、seed offsets及精确dense数量。
- 六个frozen_manifest与完整运行配置位于artifacts/rise_site_replication各run目录。
  每rank模型构造后再次写model_audit_rankN.json，实际结构不符将拒绝进入训练。
- 训练使用独立run ID，不覆盖旧权重；源码和原始参照日志SHA冻结。

## 结论与停止规则

需要分别回答：(1) Zenith是否复现HiFormer的“历史位置好、token位置差”；
(2) HyFormer的收益更依赖history encoding还是query decoding；
(3) 完整配置是否优于单位置配置，是否存在反馈标签间的反例。

如果结果不一致，应收窄为架构和位置相关的结论，而不是把不一致当作失败重跑。
不以测试结果选择一个最佳位置后替换原主表；新增结果明确为用户要求的探索性跨模型复核。
成功的完成证据必须包含scheduler COMPLETED/0:0、outcome、模型构造审计、最终Test、
样本覆盖与checkpoint，提交或进入RUNNING不等于完成。

## 新增执行入口

`scripts/run_site_replication.py`、`scripts/submit_site_replication.sbatch`、
`tests/test_site_replication.py`，均位于本研究目录；提交回执与coverage预检位于本notes目录。
已提交同一数组，后续只能检查/收集，不可重复提交整个数组。
