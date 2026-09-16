# HyFormer–MerRec 组件消融对照审计

审计日期：2026-09-16。
范围：只为新增 B、C、D 三个任务核查历史完整 SISA 对照、基础配置与数据覆盖。
本记录不包含新增训练结果，不提交或启动训练。

## 对照身份与选择依据

- 模型：HyFormer；数据集：MerRec_Action，属于规定的 4×4 接受证据范围。
- 历史完整 SISA job/task：548166_1，归档状态为 COMPLETED / 0:0。
- 历史 run ID：SISA_single_seed_HyFormer_MerRec_Action_sisa_seed20262028_ws2_bs16384_acc1_attempt1。
- seed：20262028；本轮预先固定该对照，不依据消融结果更换到其他种子。
- 原始指标日志：checkpoints/MerRec_Action/SISA_single_seed_HyFormer_MerRec_Action_sisa_seed20262028_ws2_bs16384_acc1_attempt1.log。
- 原始 Slurm 输出：logs/unirank-sisa-seed20262028-ws2-548166_1.out。
- 紧凑归档来源：experiments/sisa_paper_report/archive/results/ 中的 cell_results.csv、label_results.csv、run_evidence.csv。

标准主实验 seed 20262027 中，HyFormer–MerRec 的五任务宏平均 AUC 从
0.793737200 提升至 0.806130800，增量为 0.012393600，五个任务的 AUC 均提升；
宏平均 LogLoss 从 0.083856200 降至 0.081237000。
选择该组合是为了在已观察到较好收益的场景中分解组件贡献，不表示每个种子、每个任务均受益。
本轮直接对照 seed 20262028 的 Offer AUC 为 0.731796，低于旧基线的 0.739902；
旧基线与该对照的种子、每卡 batch 和样本覆盖不同，因此该跨轮差值只作背景记录，不能作为配对消融结论。

HyFormer 的 SISA 接入三层序列自注意力及三层 query-to-sequence 注意力，
与第一组 RankMixer–KuaiRand 的目标注意力汇聚位置不同；MerRec 的五个电商行为任务
也补充了 KuaiRand 的视频行为场景。因此该组能增加架构和任务场景的覆盖，
但两组均根据已知收益选择，不能据此推断全部 16 个组合的组件贡献。

## 历史完整 SISA 指标

下列数值从原始日志的最终 Test evaluation 段提取，已与归档 label_results.csv
逐项比较。宏平均为五任务等权算术平均，已独立重算并与 cell_results.csv 一致。

| 任务 | AUC | LogLoss |
|---|---:|---:|
| Like | 0.750689 | 0.331361 |
| Cart | 0.810189 | 0.052477 |
| Offer | 0.731796 | 0.021512 |
| Checkout | 0.855579 | 0.004954 |
| Purchase | 0.856109 | 0.003084 |
| 等权宏平均 | 0.800872400 | 0.082677600 |

## 必须保持的设置

- 2×NVIDIA H100 80GB HBM3、DDP；历史分配为 CPU 16 核、主机内存 491520 MiB。
- 每卡 batch 16384、全局 batch 32768、梯度累积 1、训练 1 epoch。
- 实验 seed 20262028，各 rank 使用 seed+rank；dataloader_seed=2027；sisa_parameter_seed=20260822。
- max_len=100、embedding_dim=16、token_dim=256、num_layers=3、num_heads=2。
- BF16 与 torch.compile 启用，gradient_checkpointing=False。
- Dense AdamW：学习率 0.0001、weight_decay=0；Sparse Adagrad：学习率 0.05。
- MerRec 运行时覆盖必须包含 sparse_optimizer_foreach=False、sparse_adagrad_chunk_size=16777216。
- 历史启动器对 MerRec 设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，新组应保持该环境。
  该项来源是 scripts/run_sisa_single_seed_task.sh 的 MerRec 启动分支，原始 INFO Params 不记录 CUDA allocator 环境。
- SISA score_dim=16、lambda_init=0.1、score_scale=1.0；各接入位置的参数种子偏移和其余默认值保持不变。
- 原任务标签顺序为 Like、Cart、Offer、Checkout、Purchase；同一数据划分、side information、drop_last=True、评估及 checkpoint 选择流程。

configs/base_configs.json 保存当前 load_config("config", "HyFormer_MerRec_Action")
返回的 60 个基础字段，不含运行时覆盖。其中 batch_size=8192、seed=20262027 是基础 YAML 值，
启动时必须覆盖为本轮 H100 对照的值，不能直接按基础 YAML 启动。
configs/reference_params.json 保存上述原始指标日志 INFO Params 的全部 74 个字段，
保持日志中的字符串表示，不仅摘录手选参数。
当前基础配置与历史 Params 的差异均对应列出的运行时设置，未发现其他基础配置漂移。

## 当前数据覆盖重算

直接构造当前 BlockedParquetBatchDataset，使用原始三个 split 的 data/user_info/item_info 路径，
distributed=True、world_size=2、batch_size=16384、drop_last=True、seed=2027；
train 开启 shuffle，valid/test 关闭 shuffle。未执行训练或遍历完整数据。

| split | rank | 分配块数 | 分配行数 | 本地批次数 | 保留行数 |
|---|---:|---:|---:|---:|---:|
| train | 0 | 4 | 72030878 | 4395 | 72007680 |
| train | 1 | 4 | 72077357 | 4397 | 72040448 |
| valid | 0 | 4 | 8258428 | 502 | 8224768 |
| valid | 1 | 4 | 8257142 | 502 | 8224768 |
| test | 0 | 4 | 5841229 | 354 | 5799936 |
| test | 1 | 4 | 5839925 | 354 | 5799936 |

六组的块数、分配行数和批次数均与历史 Slurm 日志的 blocked load balanced 记录完全一致，
记录于 configs/data_coverage.csv。保留行数为批次数乘以 16384，是当前 loader 的
drop_last 覆盖统计；不将本地批次数误写为两 rank 完全相同的优化步数。
测试集合计保留 11599872 行；验证集合计保留 16449536 行。

## 验证边界

本次核查证明配置身份、日志指标、块分配规模及 drop_last 覆盖统计对齐。
历史未保存逐样本 ID 或完整数据文件内容散列，因此同样的行数和块数不能单独证明
每个数据值、每个保留样本与历史逐字节一致。新增任务完成后仍须核对运行时配置、
数据覆盖、最终五任务指标、完成状态及异常日志，不能将提交成功写成实验完成。
原始 Params 也不包含所有软件环境和模块内部默认值；本次不能声称历史环境与当前环境逐字节相同。

本组只改变 sisa_variant 和区分输出的 model_id；B、C、D 均从头训练。
单种子消融可以给出本组的描述性组件贡献，不报告跨种子标准差、统计显著性或普遍协同结论。
