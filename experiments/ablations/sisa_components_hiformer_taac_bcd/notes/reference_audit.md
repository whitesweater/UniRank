# HiFormer–TAAC-25 组件消融对照审计

审计日期：2026-09-16。
范围：为第三组 B、C、D 三个新增任务核查历史完整 SISA 对照、基础配置和数据覆盖。
本记录不包含新增训练结果；审计步骤未启动训练。

## 对照身份与选择依据

- 模型：HiFormer；数据集：TencentGR_10M_Action（TAAC-25），属于规定的 4×4 接受证据范围。
- 历史完整 SISA job/task：548166_14，归档状态为 COMPLETED / 0:0。
- 历史 run ID：SISA_single_seed_HiFormer_TencentGR_10M_Action_sisa_seed20262028_ws2_bs16384_acc1_attempt1。
- seed：20262028；本轮预先固定该对照，不依据消融结果选择其他种子。
- 原始指标日志：checkpoints/TencentGR_10M_Action/SISA_single_seed_HiFormer_TencentGR_10M_Action_sisa_seed20262028_ws2_bs16384_acc1_attempt1.log。
- 原始 Slurm 输出：logs/unirank-sisa-seed20262028-ws2-548166_14.out。
- 紧凑归档来源：experiments/sisa_paper_report/archive/results/ 中的 cell_results.csv、label_results.csv、run_evidence.csv。

标准主实验 seed 20262027 中，该组合两任务宏平均 AUC 从 0.855235000
提升至 0.866796000，增量为 0.011561000；点击和转化两个任务的 AUC 均提升。
宏平均 LogLoss 从 0.032188500 降至 0.027953500，改善 0.004235000。
选择该组合是为了在已观察到较好收益的场景中分解组件贡献；
该选择基于已有结果，不能据此推断全部 16 个组合的组件贡献。

相比前两组 RankMixer–KuaiRand 和 HyFormer–MerRec，本组增加 HiFormer
骨干及 TAAC-25 点击、转化任务的覆盖。HiFormer 的 SISA 同时作用于
目标注意力汇聚和三层特征融合 token 注意力。后者的 token 表示融合后的特征与任务，
不是按时间发生的行为事件；虽然配置中的衰减参考名为 sequence_end，
也不能据此把该位置解释为时间遗忘。本轮同时消融全部既有接入位置，
因此只能分析组件的总体贡献，不能区分收益来自历史行为汇聚还是特征融合位置。

## 历史完整 SISA 指标

从原始日志的最终 Test evaluation 段提取下列指标，与归档 label_results.csv
逐项核对一致。宏平均按两任务等权计算，独立重算后与 cell_results.csv 一致。

| 任务 | AUC | LogLoss |
|---|---:|---:|
| is_click | 0.851302 | 0.043410 |
| is_conversion | 0.890414 | 0.017180 |
| 等权宏平均 | 0.870858000 | 0.030295000 |

旧四卡 baseline 和本轮两卡对照的种子、每卡 batch 与样本覆盖协议不同，
旧基线仅用于上述主实验选择依据，不作为本组三个消融任务的严格配对对照。

## 固定协议及运行时覆盖

- 2×NVIDIA H100 80GB HBM3、DDP；历史分配 CPU 16 核、主机内存 491520 MiB。
- 每卡 batch 16384、全局 batch 32768、梯度累积 1、训练 1 epoch。
- 实验 seed 20262028，各 rank 使用 seed+rank；dataloader_seed=2027；sisa_parameter_seed=20260822。
- max_len=100、embedding_dim=16、token_dim=256、num_layers=3、num_heads=2、qkv_rank=256、expansion_factor=2。
- BF16 与 torch.compile 启用；gradient_checkpointing=False；attention_dropout=0、net_dropout=0。
- Dense AdamW：学习率 0.0001、weight_decay=0；Sparse Adagrad：学习率 0.05。
- 历史优化器日志明确为 sparse foreach=None；没有 sparse_optimizer_foreach 或
  sparse_adagrad_chunk_size 的运行时覆盖，本组保持默认路径。
- 历史启动器只在 MerRec 分支设置 expandable_segments:True；本组不主动增加该 allocator 特殊覆盖。
  历史日志未保存完整环境变量快照，因此不据此断言历史进程没有继承其他 allocator 环境。
- SISA score_dim=16、lambda_init=0.1、score_scale=1.0；各接入位置的参考方式与参数种子偏移沿用主实验。
- group_id=user_index；任务顺序 is_click、is_conversion；相同数据路径、side information、特征、损失、
  drop_last=True、checkpoint 选择、验证和测试流程。
- TAAC-25 的 106、107、108、110 特征各自配置 max_len=5，这是特征池化长度，
  与行为历史 max_len=100 的含义不同；全部按原配置保留。

configs/base_configs.json 保存当前 load_config("config", "HiFormer_TencentGR_10M_Action")
返回的全部 60 个基础字段。其中 batch_size=8192、seed=20262027 是 YAML 基础值，
须通过启动器覆盖为该 H100 对照的 batch_size=16384、seed=20262028。
configs/reference_params.json 保存原始 INFO Params 的全部 72 个字段及其字符串表示。
基础配置与原日志的差异均为上述已知运行时覆盖，未发现其他基础配置漂移。

## 当前数据覆盖重算

直接构造当前 BlockedParquetBatchDataset，使用三个 split 原配置的 data/user_info/item_info 路径，
distributed=True、world_size=2、batch_size=16384、drop_last=True、seed=2027；
train 开启 shuffle，valid/test 关闭 shuffle。未执行训练或遍历完整数据。

| split | rank | 分配块数 | 分配行数 | 本地批次数 | 保留行数 |
|---|---:|---:|---:|---:|---:|
| train | 0 | 8 | 293060408 | 17881 | 292962304 |
| train | 1 | 8 | 293057086 | 17883 | 292995072 |
| valid | 0 | 4 | 8922909 | 543 | 8896512 |
| valid | 1 | 4 | 8923231 | 543 | 8896512 |
| test | 0 | 4 | 76624928 | 4674 | 76578816 |
| test | 1 | 4 | 76618584 | 4674 | 76578816 |

六组块数、分配行数和批次数均与 548166_14 的 blocked load balanced 日志逐项匹配，
记录于 configs/data_coverage.csv。保留行数为批次数乘以 16384，是 loader 的
drop_last 覆盖统计，不应把不相等的本地批次数写成完全相同的两 rank 优化步数。
验证集合计保留 17793024 行，测试集合计保留 153157632 行。

## 验证边界

历史记录未保存逐样本 ID 和完整数据文件内容散列，因此相同块数、行数及批次数
不能单独证明样本值与历史逐字节一致。Params 也不覆盖全部软件环境与内部默认值；
本次审计不声称历史和当前运行环境逐字节相同。
训练完成后仍需检查实际配置、两 rank 数据覆盖、最终两任务指标及异常日志。

B、C、D 都从头训练，只改变 sisa_variant 和区分输出的 model_id，
保持内容投影、共有参数初始化、每头缩放、骨干、接入位置及已有掩码处理。
本轮单种子结果仅用于选定组合的描述性组件分析，不报告跨种子标准差或统计显著性。
没有接入位置消融及固定位置规则对照，不能据此声称收益全部来自时间衰减，
也不能单独证明内容自适应优于固定位置规则。
