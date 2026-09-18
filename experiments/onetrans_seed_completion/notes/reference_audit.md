# OneTrans 三种子补齐：历史参考协议审计

审计日期：2026-09-18。本记录冻结旧 seed 20262027 的五数据集 baseline/SISA 共十次正式运行，供后续补齐 seed 20262028、20262029 使用。
用户已确认仅补齐 SISA：五个数据集各新增 seed20262028、seed20262029，共十个任务，不新增 baseline。旧 baseline 五次运行继续作为历史对照；本审计只核查归档，不启动训练。
OneTrans 及 Taobao 不属于现行 SISA 论文接受的 4×4 主实验范围，本独立补齐研究的结果不得自动混入该范围的汇总。

## 优先发现与处置要求

1. TAAC-25 的当前基础配置 group_id=user_index，历史两个正式 OneTrans 运行均为 group_id=user_id。这是实际配置差异，需要在新 study 中显式处理并审计；若严格匹配旧设置，应恢复旧 group_id，而不能把当前 YAML 原值宣称为旧实验原值。该字段在当前评估代码中会用于收集 group_id，即使最终仅报告普通 AUC 与 LogLoss，也不能隐去漂移。
2. QK-Video、Taobao、MerRec 的旧日志记录了旧机器数据根目录绝对路径；当前数据位于仓库相对 ./datasets。KuaiRand、TAAC-25 旧日志已经使用相对数据路径。路径迁移应按数据集内后缀映射，不能直接复用旧机器前缀。当前五数据集的四 rank 块分配与历史均一致，详见下文。
3. OneTrans–MerRec 的旧正式 baseline/SISA 没有 sparse_optimizer_foreach 或 sparse_adagrad_chunk_size 覆盖。后续其他模型两卡复跑使用过的 foreach=False、chunk_size=16777216，不能直接带入本轮并仍称为与旧 OneTrans 协议相同。
4. 旧十份日志均没有显式 dataloader_seed、sisa_parameter_seed 字段或 RNG_PROTOCOL 行。旧实现可说明默认子随机流，但不能把推定值写成当时日志显式核验的事实。
5. 除旧数据路径、TAAC-25 的 group_id，以及 DDP/BF16/SISA 开关等已知运行时覆盖外，当前基础配置与历史参数未发现其他差异。

## 原始参考运行

输入源为 experiments/sisa_native_strict/results/runs.csv 的八个 OneTrans 行，以及 experiments/sisa_expansion_acd/results/runs.csv 的两个 OneTrans–TAAC-25 行。
每个 checkpoint 日志与对应 Slurm 日志中的唯一 INFO Params 会话完全一致；每个最终 Test evaluation 段任务齐全，AUC 与 LogLoss 逐项匹配旧 runs.csv 的 metrics_json。
原始 strict 八行的成功状态依据是完成标记，不是现时 sacct 返回；reference_runs.csv 原样保留 COMPLETED_MARKER 与 success_marker 证据类型，不擅自改写成账务系统核实的 COMPLETED。
两个 expansion 行保留原有 COMPLETED/0:0 与 sacct 证据类型。本审计验证了原始 SISA_EXPANSION_COMPLETE 标记，不声称重新查询了历史账务状态。

| 数据集 | setting | job/task | seed | 旧 GPU 类型 | Params 字段数 | 宏 AUC | 宏 LogLoss |
|---|---|---|---:|---|---:|---:|---:|
| QK_Video_Action | baseline | 14810_0 | 20262027 | NVIDIA GeForce RTX 4090 | 67 | 0.933617750 | 0.073827250 |
| QK_Video_Action | sisa | 14810_1 | 20262027 | NVIDIA GeForce RTX 4090 | 71 | 0.934054500 | 0.073634000 |
| KuaiRand_Video_Action | baseline | 14811_2 | 20262027 | NVIDIA L40S | 68 | 0.857342167 | 0.186336833 |
| KuaiRand_Video_Action | sisa | 14811_3 | 20262027 | NVIDIA L40S | 72 | 0.865032333 | 0.177600833 |
| Taobao_Action | baseline | 14806_4 | 20262027 | NVIDIA GeForce RTX 4090 | 67 | 0.729981750 | 0.057353500 |
| Taobao_Action | sisa | 14810_5 | 20262027 | NVIDIA GeForce RTX 4090 | 71 | 0.726185750 | 0.057480250 |
| MerRec_Action | baseline | 14811_6 | 20262027 | NVIDIA L40S | 68 | 0.808730400 | 0.082217000 |
| MerRec_Action | sisa | 14811_7 | 20262027 | NVIDIA L40S | 72 | 0.817875000 | 0.080254200 |
| TencentGR_10M_Action | baseline | 543887_0 | 20262027 | NVIDIA H100 80GB HBM3 | 67 | 0.855623500 | 0.033613000 |
| TencentGR_10M_Action | sisa | 543257_1 | 20262027 | NVIDIA H100 80GB HBM3 | 71 | 0.860746500 | 0.033401500 |

strict 的 GPU 名称由旧 runs.csv 的 gpu_type 分类转写为通用设备名称；TAAC-25 的 H100 由原始 Slurm ACD_EXPANSION_HARDWARE 行验证。
参考 run ID、原始 checkpoint/Slurm 相对路径、源表路径及完整原始 metrics_json 均保存在 configs/reference_runs.csv。

## 固定设置与随机流语义

十次旧运行均为 world_size=4、每卡 batch=8192、全局 batch=32768、梯度累积1、1 epoch、max_len=100；BF16 与 torch.compile 开启，gradient_checkpointing=False。
稠密优化器 AdamW，学习率0.0001、weight_decay=0；稀疏 Adagrad，学习率0.05；各数据集标签、特征和损失列表保持其原值。
SISA 旧正式设置为 enabled=True、score_dim=16、lambda_init=0.1、score_scale=1.0；baseline 没有启用 SISA。
新运行计划使用4×H100，但继续四rank、每卡8192以保持块分配与drop_last覆盖。旧GPU差异应记录，不能声称跨硬件逐位数值相同。CPU或主机内存资源的增加不应被误写为训练预算增加。

仓库所存历史实现 1217a70（2026-08-24）明确使用 seed_everything(params["seed"]+rank)，loader 使用固定 2026+epoch+rank，SISA 使用隔离 parameter_seed 默认20260821并叠加接入位置偏移。
其中 strict 实际运行于2026-08-22，早于该已存提交，因此这些随机流只能表述为依据所存旧实现推定；旧日志自身只明确实验seed=20262027。
当前实现支持显式子seed；若补齐方案采用 seed20262028/29 对应 loader2027/2028、SISA20260822/23，应在任务表与运行日志明确这一派生规则，并将其作为种子变化的一部分。
不能将各rank的seed+rank或者不同内部子流当作额外独立实验种子。

## 数据分配核查

configs/data_coverage.csv 包含十个旧运行×三个split×四个rank共120行，列为dataset、setting、split、rank、blocks、assigned_rows、batches、processed_rows。
各dataset的baseline与SISA逐rank覆盖完全相同。对当前真实数据，使用BlockedParquetBatchDataset（world_size=4、batch_size=8192、drop_last=True、seed=2026）重建每个split；通过其当前贪心块分配方法生成四rank结果。
五数据集×三个split×四rank的60个独立分配单元，块数、分配行数和完整批次数全部与历史两个setting一致。随机子seed影响训练块顺序和块内顺序，不改变这些块分配计数。

| 数据集 | train保留行数 | valid保留行数 | test保留行数 |
|---|---:|---:|---:|
| QK_Video_Action | 394526720 | 50053120 | 48562176 |
| KuaiRand_Video_Action | 255959040 | 34308096 | 32980992 |
| Taobao_Action | 17702912 | 2842624 | 2883584 |
| MerRec_Action | 144072704 | 16482304 | 11640832 |
| TencentGR_10M_Action | 586047488 | 17809408 | 153206784 |

processed_rows=每rank完整批次数×8192，属于drop_last覆盖统计，不是独立用户数。新运行结束后仍须对比实际日志。
历史没有全量数据文件内容散列或逐样本ID清单，因此相同块数和行数不能独立证明每个数据值、每个被保留样本与旧环境逐字节一致。
若新种子改变块内shuffle，则被丢弃的训练尾部样本也可能变化；不能把覆盖计数相同写成不同seed逐样本顺序相同。

## 冻结文件与提交边界

- configs/reference_params.json：以 OneTrans_<dataset>_<setting> 为key，保留全部原始INFO Params字段及字符串表示；仅按下述规则归一运行路径，不宣称路径字符串仍是原始值。
- configs/base_configs.json：以 OneTrans_<dataset> 为key，保存本次load_config基础值，不含运行时覆盖；TAAC-25的当前group_id=user_index仍如实保留。
- configs/reference_runs.csv：十个历史参考的身份、成功证据、协议、原始相对日志路径和指标。
- configs/data_coverage.csv：120行旧协议覆盖，已经当前数据重建验证。

原样参考快照已保存至由Git忽略的 artifacts/onetrans_seed_completion/reference_params_raw.json；可提交的 configs/reference_params.json 不含历史机器绝对路径。只允许如下归一：data_root 设为 ./datasets；model_root 设为 ./checkpoints/；train/valid/test 的 data、user_info、item_info 路径按 dataset_id 和原有 split/类型后缀映射到 ./datasets/<dataset>/<split>/<类型>。每个原路径先检查对应后缀，再替换机器前缀；未改动任何非路径参数。TAAC-25 的 reference group_id 继续保留旧 user_id，当前 base_configs 的 user_index 也原样保留，差异由启动方案明确处理。
本审计未修改模型代码、训练脚本、现有报告或用户未提交的 report.md，未执行Git提交和训练提交。

## 原始来源散列与归一审计

散列算法为 SHA-256。以下散列由本次直接读取的原始字节计算，可用于核对原始快照及参考日志，没有把归一内容当作原始日志。

- 原始十运行 Params 快照：fa844fdae185302eb63c4035ddc2eeccb43ac0fbb74662fcc92526f1facfae76。
- 便携 Params 快照：ce014788be8029a2a92f8c09aa2f23f08aac12cd5fcb8638e04131b4303be3b3。
- 实际改变路径字符串 60 个，涉及 6 个运行；其他运行的路径本来已采用相同相对形式。所有运行的非路径字段逐项保持不变。

| 数据集 | setting | 来源类型 | 相对日志路径 | SHA-256 |
|---|---|---|---|---|
| QK_Video_Action | baseline | checkpoint_log | checkpoints/QK_Video_Action/SISA_strict_OneTrans_QK_Video_Action_baseline_seed20262027.log | bbbda0d9eea09a750aca9257949ac49a4257a9b32890a0f3dfc8804b94f7a527 |
| QK_Video_Action | baseline | slurm_log | logs/unirank-sisa-strict-14810_0.out | 52c68a61c9dbda74f0aab771ee8716d029b2ba5dc5b71c94f5620190691617e2 |
| QK_Video_Action | sisa | checkpoint_log | checkpoints/QK_Video_Action/SISA_strict_OneTrans_QK_Video_Action_sisa_seed20262027.log | afc5de04bb8ffefd7755d410ff50381993df810131ff73143aac357372b920aa |
| QK_Video_Action | sisa | slurm_log | logs/unirank-sisa-strict-14810_1.out | 34534ae2b62a84b5b2dd180bb8f43438aabb4fcc59dc72a52873db166f331b30 |
| KuaiRand_Video_Action | baseline | checkpoint_log | checkpoints/KuaiRand_Video_Action/SISA_strict_OneTrans_KuaiRand_Video_Action_baseline_seed20262027.log | fcb7e63b69a5f5154791486c4c33dab87a0235676873fc1eb7be5d32aafc4851 |
| KuaiRand_Video_Action | baseline | slurm_log | logs/unirank-sisa-strict-14811_2.out | 20ff500c2154dd8ad007f1d7d0714cff2870ed27a9287b7b254b89131ba59b37 |
| KuaiRand_Video_Action | sisa | checkpoint_log | checkpoints/KuaiRand_Video_Action/SISA_strict_OneTrans_KuaiRand_Video_Action_sisa_seed20262027.log | b6821e6fd15c05711182bb1dd6ff3c366da4973c9a7e5e589e76e61b36f2edca |
| KuaiRand_Video_Action | sisa | slurm_log | logs/unirank-sisa-strict-14811_3.out | e7907b67a1af455b02f587ccb7124db995a5dc9d9036096874727a68c5104650 |
| Taobao_Action | baseline | checkpoint_log | checkpoints/Taobao_Action/SISA_strict_OneTrans_Taobao_Action_baseline_seed20262027.log | dd9d9fb5bfee0bae02e9d011a5194efd09a7d6fe0e49e55da6816ea85868bd26 |
| Taobao_Action | baseline | slurm_log | logs/unirank-sisa-strict-14806_4.out | 924415d6ae7249abd375adaee947e9a5354d12757dd2b0963bea5331b68ceb61 |
| Taobao_Action | sisa | checkpoint_log | checkpoints/Taobao_Action/SISA_strict_OneTrans_Taobao_Action_sisa_seed20262027.log | e88aa7a4080e75ff9a04208484917d7ffb5a397b3b0cf1ead2c8ff2c8d94e875 |
| Taobao_Action | sisa | slurm_log | logs/unirank-sisa-strict-14810_5.out | 7a4006b859b68c8af422d06c69db0a5e672aee8713f8ed09c0a1bd4b614f6bc9 |
| MerRec_Action | baseline | checkpoint_log | checkpoints/MerRec_Action/SISA_strict_OneTrans_MerRec_Action_baseline_seed20262027.log | aecdd65d36004c22998dc7a296732bc45135a7909482dd42ef29138771004e08 |
| MerRec_Action | baseline | slurm_log | logs/unirank-sisa-strict-14811_6.out | ccb655a57b806ba45cf00da228fee120c3c855f9ed5d38d1e8ef3764285868e4 |
| MerRec_Action | sisa | checkpoint_log | checkpoints/MerRec_Action/SISA_strict_OneTrans_MerRec_Action_sisa_seed20262027.log | fdad36af73f19778458b335de1f2339ead65e02fc2ad1335fc9c2998b8413b3b |
| MerRec_Action | sisa | slurm_log | logs/unirank-sisa-strict-14811_7.out | b6131818d0273eb4c62cd4320942ba6073bc7a4f4c902078d69f7f06834f62a1 |
| TencentGR_10M_Action | baseline | checkpoint_log | checkpoints/TencentGR_10M_Action/SISA_expansion_OneTrans_TencentGR_10M_Action_baseline_seed20262027.log | 65fea3714bc977129ad09a1ae6f89be7afa9df037de9f53caec2b0970b950bc9 |
| TencentGR_10M_Action | baseline | slurm_log | logs/unirank-sisa-expansion-acd-543887_0.out | 7be715fb173c015d871fa0084f4cf909b828bdc8f08bb87a89725ada39c26ade |
| TencentGR_10M_Action | sisa | checkpoint_log | checkpoints/TencentGR_10M_Action/SISA_expansion_OneTrans_TencentGR_10M_Action_sisa_seed20262027.log | 552f9089e78b1a2c184d8f57f18f9f1cd50ed4b6ec1c682a05e450bac7228d1b |
| TencentGR_10M_Action | sisa | slurm_log | logs/unirank-sisa-expansion-acd-543257_1.out | 197692c2f1baa6f1293547676eec9b8a963425600632df722268a1f23d45f792 |
