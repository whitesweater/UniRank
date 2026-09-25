# 论文图表设计与证据地图

表面：独立静态 PDF/SVG，PNG 用于视觉核验；属于第三方论文图，不添加产品品牌装饰。
绘图环境单独由 `figure_env/pyproject.toml`、`uv.lock` 管理，不改变训练环境。

| 文件 | 研究问题 | 数据/比较 | 形式 | 完成与限制 |
|---|---|---|---|---|
| component_effect_by_history | AM/PA条件收益是否随可见长度变化 | 同种子B/C/D/E，五个互斥长度组 | 三标记系列与用户簇区间 | 每个点对五任务等权；B非原生Baseline |
| component_effect_by_frequency | 收益是否仅集中于稀疏或高频实体 | E−B，训练目标item/user频次 | 并排两幅点区间图，共用纵轴 | zero只称训练目标未见，不能直接称冷启动 |
| task_effect_by_history | 宏平均是否掩盖任务差异 | E−B，五任务×五长度组 | 有符号热图，零中心蓝/金 | 点估计；全部区间、支持量在CSV，不以星号筛选 |
| conditional_component_contrasts | Offer上的E−D是否集中于短历史；宏交互项如何随组变化 | 完整模型相对PA-only，以及E−C−D+B | 两幅点区间图、明确零线 | 名义区间；训练后对比不能分离容量，不把负交互称因果冲突 |

当前图只用正式643839导出、643869聚合得到的真实数值，保留负值和空历史组。
不把统计区间解释为训练随机性，不将样本组AUC按样本量加权成总体AUC。
后续原生Baseline对照、跨组AUC分解、内部机制与运行成本分别生成独立图，
不通过换标签把已有Content-only对照包装成原生Baseline。

运行：`uv run --project experiments/rise_sample_analysis/figure_env --locked python experiments/rise_sample_analysis/scripts/plot_component_analysis.py`。
每次更新数据后应重新生成并检查导出图；生成manifest中的视觉核验状态需以实际查看结果更新。
