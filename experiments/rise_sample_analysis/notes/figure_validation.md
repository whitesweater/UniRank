# 科学图第一批视觉核验

四幅图均已检查实际PNG输出，PDF/SVG来自同一Matplotlib figure。历史长度图初版标题与纵向图例接近，已改为横向三系列图例并重新检查。

- 历史长度：五组支持量、三种标记、零线及95%用户簇区间可读；B明确Content-only。
- 训练频次：两面板共享纵轴，频次0明确仅为训练目标未见；分组比例和整数边界齐全。
- 任务热图：保留Offer空历史负值，符号和零中心配色正确；无显著性星号。
- 条件组件：Offer的E−D与宏交互项分开，零参考线和各自坐标尺度明确，不暗示容量/机制纯因果识别。

图表来自643839完整配对预测和643869聚合，原生Baseline结果另行制作。初版聚合Logloss的float32/float64边界检查见results/group_analysis_v1_logloss_reconciliation.json；当前20项观测Logloss均与官方口径相符，AUC图不受影响。

绘图环境独立于训练环境，依赖锁定在figure_env/uv.lock。图表生成器保留输入及自身SHA，当前导出文件SHA记录于figures/component_figures_manifest.json。这不替代最终论文PDF排版检查。
