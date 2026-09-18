# OneTrans SISA 三种子补齐

按用户最新选择，仅补 SISA，覆盖 QK-Video、KuaiRand、TAAC-25、MerRec
四个数据集。复用已有 seed 20262027，为每个数据集新增 20262028、20262029，
共保留八个任务；不重跑 baseline，也不重复已有 seed。完成后每个数据集有三次 SISA。
最初提交的十个任务中，Taobao 的 630948_6、630948_7 已按用户要求取消；保留
原任务编号与旧结果供审计，启动器禁止重新启动这两个编号，未来默认提交也排除它们。

本研究独立归档。OneTrans 及 Taobao 不在当前论文接受的四模型×四数据集证据范围内，
本次用户授权的补跑结果不自动混入该主实验汇总。

## 固定协议

- 新任务全部使用 4×H100，DDP world size=4，每卡 batch=8192，全局 batch=32768。
- 保持旧 OneTrans 的 1 epoch、最大历史长度100、梯度累积1、BF16、torch.compile，
  梯度检查点关闭，稠密 AdamW 学习率0.0001、稀疏 Adagrad 学习率0.05。
- 沿用旧特征、标签、损失、数据划分、四rank块分配、drop_last、验证与测试协议。
- 完整 SISA：score_dim=16、lambda_init=0.1、score_scale=1.0；不使用组件消融。
- 实验种子20262028/20262029；各rank为seed+rank；loader子seed分别2027/2028，
  SISA参数子seed分别20260822/20260823；子流不计为额外实验seed。
- 旧seed27来自已有四卡运行，旧GPU型号按数据集为RTX4090、L40S或H100，
  新轮统一H100。保持四rank及每卡batch以避免原先两卡复跑引入的尾批覆盖差异；
  跨GPU数值差异仍可能存在，三个点的方差不能解释为纯随机种子方差。
- 每任务申请32 CPU、960 GiB主机内存、12小时，最多四任务并行，不自动重试或改变预算。

## 本轮专用配置与启动前审计

configs/base_configs.json 保存当前仓库的五份合并配置；configs/runtime 保存实际训练
使用的可移植冻结配置。TAAC-25 的group_id在当前仓库为user_index，历史OneTrans
为user_id，因此只在本study专用配置中恢复user_id。当前Parquet数据与特征配置均
包含user_id，核心训练/评估实现不变。主配置文件保持不动。

历史路径按相同数据集及split后缀映射为 ./datasets 下相对路径。便携历史参数位于
configs/reference_params.json；原样历史路径快照放在Git忽略的artifacts目录，原始
日志及快照散列记录在notes/reference_audit.md。历史十次baseline/SISA均保留用于审计。

旧OneTrans–MerRec没有后续其他模型两卡复跑的foreach/chunk_size特殊覆盖，本轮
不增加这些覆盖。configs/data_coverage.csv保存旧两种setting的四rank覆盖；启动前
已重算当前五数据集train/valid/test共60个独立分配单元，全部匹配历史记录。

启动器检查当前主配置未漂移、专用冻结配置完整匹配历史参数；除run ID、实验seed
及明确的子seed外不允许新的有效参数差异。已有日志、checkpoint或artifact目录会
阻止覆盖；需要重试时必须增加attempt并在采集器明确选择，不能按测试指标挑重试。

## 原任务映射与当前保留范围

| task_id | 数据集 | seed |
|---|---|---:|
| 0 | QK-Video | 20262028 |
| 1 | QK-Video | 20262029 |
| 2 | KuaiRand | 20262028 |
| 3 | KuaiRand | 20262029 |
| 4 | TAAC-25 | 20262028 |
| 5 | TAAC-25 | 20262029 |
| 6（已取消） | Taobao | 20262028 |
| 7（已取消） | Taobao | 20262029 |
| 8 | MerRec | 20262028 |
| 9 | MerRec | 20262029 |

精确run ID、初始化子seed与attempt见configs/tasks.csv。启动脚本为
scripts/submit_onetrans_seed_completion.sbatch，实际提交记录在notes/submission.json，
取消回执在notes/cancellation.json。configs/tasks.csv保留最初十任务映射，不重排编号。
每次运行在artifacts/onetrans_seed_completion下保存manifest和outcome，包含源码散列、
有效参数、命令、GPU、作业标识与返回码；原始日志和模型分别在logs及checkpoints下。

## 结果口径

只从成功完成并通过配置、覆盖和双日志核验的最终Test段取AUC与LogLoss。
结果收集器为scripts/collect_onetrans_seed_completion.py，紧凑证据保存在results。
未完成、取消、失败或不一致的运行保留状态且指标留空；仅三次SISA均合格时生成均值与
样本标准差（ddof=1），同时保留每个标签和每轮的结果。

baseline仅有seed20262027，不把一个baseline复制成三个独立baseline；本研究不支持
三种子配对增益或配对显著性检验。历史同seed27的baseline/SISA差值仍可单独描述，
SISA三次均值与单次baseline的差值只能作为描述性补充。Taobao此次不补齐，不计算
其三seed均值。刷新命令为 `.venv/bin/python scripts/collect_onetrans_seed_completion.py`。
