# SISA 组件消融：HyFormer–MerRec

新增第二个模型—数据集组合：HyFormer–MerRec_Action。只训练 B/C/D 三个任务，
seed 固定为 20262028；完整 SISA E 复用已完成的 H100 任务 548166_1。
第一组 RankMixer–KuaiRand 的任务和配置保留，不重复提交。

## 研究问题与选择依据

检验衰减与旋转在另一种模型结构及电商行为数据中的独立和组合贡献。
主实验 seed 20262027 中，该组合五任务 AUC 均提升，任务宏平均 AUC 增量为
0.0123936；宏平均 LogLoss 改善 0.0026192。该已知收益用于选择机制分析场景，
本轮结论仅覆盖选定组合，不能据此代表全部 16 个标准模型—数据集组合。

本轮直接对照选定 H100 首轮 seed 20262028，不按消融结果更换对照：
宏平均 AUC 为 0.8008724，宏平均 LogLoss 为 0.0826776。
原始四卡 baseline 不作为本轮严格配对对照；不新增 A/E 训练。

## 新增任务与唯一实验变量

| 数组任务 ID | 组别 | 配置值 | 内容打分 | 动态衰减 | 动态旋转 |
|---|---|---|---|---|---|
| 0 | B | content_only | 有 | 无 | 无 |
| 1 | C | decay_only | 有 | 有 | 无 |
| 2 | D | rotation_only | 有 | 无 | 有 |

三组均从头训练，在原完整 SISA 的所有 HyFormer 接入位置应用同一组件模式。
接入位置和对应衰减参考、内容投影、每头可学习缩放、原始骨干和已有掩码处理均保留。
关闭衰减为 log(alpha)=0，关闭旋转为相位为零。共有参数使用相同隔离 RNG
初始化；未启用分支被冻结且不参与前向计算，不把冻结参数计为有效容量匹配。

## 与 H100 对照完全匹配的设置

- 对照任务：548166_1。
- 对照 run ID：SISA_single_seed_HyFormer_MerRec_Action_sisa_seed20262028_ws2_bs16384_acc1_attempt1。
- 实验 seed 20262028，rank 使用 seed+rank；数据加载 seed 2027；SISA 参数 seed 20260822。
- 2×H100，每卡 batch 16384，全局 batch 32768，梯度累积 1。
- 训练 1 epoch，最大序列长度 100，BF16 与 torch.compile 启用。
- SISA score_dim=16、lambda_init=0.1、score_scale=1.0；其余默认值沿用对照。
- MerRec 保留 sparse_optimizer_foreach=False、sparse_adagrad_chunk_size=16777216，
  并沿用 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，避免引入优化器路径差异。
- CPU 16 核、主机内存 480 GiB；数据、特征、任务标签、优化器、学习率、drop_last、
  验证和测试流程全部匹配对应 H100 轮次。

configs/base_configs.json 保存基础 YAML，reference_params.json 保存原始日志全部配置。
configs/protocol_comparison.csv 对比三组最终有效配置，允许差异仅为 sisa_variant
和用于区分输出的 model_id；其他差异在启动前报错。
configs/data_coverage.csv 比对当前数据在两个 rank 的 train/valid/test 块数、行数和批次数。
覆盖检查不等同于对历史数据进行逐字节哈希核验；正式结束后仍须检查运行日志的实际覆盖。

## 评估、归档和解释边界

报告 Like、Cart、Offer、Checkout、Purchase 的 AUC 与 LogLoss，以及等权任务宏平均。
比较 C−B、D−B、E−C、E−D；AUC 交互量为 E−C−D+B。
只有一个种子，不宣称统计显著性；不根据预期效果省略退化项。
没有固定规则对照，本轮不能单独证明内容自适应优于固定位置规则。

启动脚本：scripts/submit_sisa_components_hyformer_merrec_bcd.sbatch。
共享启动器：scripts/run_sisa_component_ablation.py --study sisa_components_hyformer_merrec_bcd。
任务映射：configs/tasks.csv；提交和验证记录：notes/launch.md。
数组固定 0–2，最多两个任务并行。失败重试增加 attempt，禁止覆盖已有日志和模型。
原始输出保存在 logs/、checkpoints/、artifacts/sisa_components_hyformer_merrec_bcd/，均由 Git 忽略。
