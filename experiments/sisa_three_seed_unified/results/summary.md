# 论文 / baseline / 三-seed SISA 统一结果

SISA 三次实验 seed 为 **20262027、20262028、20262029**。内部 dataloader seed 和 SISA parameter seed 只是各轮 RNG 子流，不计作额外实验 seed。

- 对齐范围：**16** 个模型×数据集单元、**68** 个标签。
- 三-seed SISA 相对本地 baseline 的 cell-macro ΔAUC：**+0.006415**；正向单元 **15/16**。
- 三-seed SISA 相对论文表值的 cell-macro ΔAUC：**+0.006817**；正向单元 **15/16**。
- 三-seed SISA 相对 baseline 的 cell-macro Δlogloss：**-0.001513** （负值更好）。
- 三轮最优 AUC 模型完全一致：**8/17** 个数据集×标签任务。

seed 20262027 使用 4 卡混合硬件和每卡 batch 8192；seed 20262028/20262029 使用 2×H100 和每卡 batch 16384。三点均值、样本标准差和范围只能作为探索性描述，不能解释为纯 seed 方差或显著性检验。
