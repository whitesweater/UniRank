# SISA 投影维度敏感性实验

状态：2026-09-23 03:26（北京时间）8、32 维完成并通过审计；64 维首批 OOM，后续处置待定。
核查日期：2026-09-23。

## 研究问题

固定完整 SISA，考察每头投影维度对预测效果与参数开销的影响。当前代码和既有
组件消融的完整 SISA 对照均使用 `sisa_score_dim=16`。建议扫描
`r ∈ {8, 16, 32, 64}`；这是待运行方案，不是已验证的最佳取值范围。
用户已明确排除 4 维训练。

输入隐表示分别通过 B、C 投影得到每个注意力头的 r 维内容通道。完整 SISA
在这些通道上施加累计衰减和旋转，再形成 softmax 前的附加注意力分数。
`sisa_score_dim` 表示每头的通道数，头数由骨干原有配置决定；H 个头的单个
B 或 C 投影总输出宽度为 H×r。旋转按二维通道对实现，因此 r 必须为正偶数，
每头对应 r/2 个旋转平面。衰减投影仍然是每个 token、每头一个标量。

实现位置为 `unirank/pytorch/layers/attentions/sisa.py`；训练入口已经支持
`run_expid.py --sisa-score-dim`，本轮无需修改机制实现。

## 实验范围与任务数量

仅 HyFormer–MerRec_Action，与当前组件消融正文的展示范围一致。

形成四个维度点。16 维复用预先指定并已重新审计的完整 SISA 对照，新增
8、32、64 维三个从头训练任务。本轮固定 `sisa_variant=full`。

| 每头维度 r | 每头旋转平面数 | 建议处置 |
|---:|---:|---|
| 8 | 4 | 新增训练 |
| 16 | 8 | 审计后复用完整 SISA 对照 |
| 32 | 16 | 新增训练 |
| 64 | 32 | 新增训练 |

## HyFormer–MerRec 的具体对照

预指定对照为历史完整 SISA 任务 `548166_1`，run ID 为
`SISA_single_seed_HyFormer_MerRec_Action_sisa_seed20262028_ws2_bs16384_acc1_attempt1`。
2026-09-23 已重新读取原始 checkpoint 日志：其 Params 与既有冻结
`reference_params.json` 完全一致；当前基础配置通过组件启动器的历史协议校验。
最终 Test evaluation 覆盖 Like、Cart、Offer、Checkout、Purchase 五个标签，
重新计算的等权宏平均 AUC 为 0.8008724、LogLoss 为 0.0826776。
以上为预指定的 16 维历史对照指标，新增结果见下表。

## 当前已验证结果

| 每头维度 | 状态 | 五任务宏平均 AUC | 五任务宏平均 LogLoss | 相对 16 维 AUC 差值 | 相对 16 维 LogLoss 差值 |
|---:|---|---:|---:|---:|---:|
| 8 | 完成并通过审计 | 0.7970614 | 0.0831128 | -0.0038110 | +0.0004352 |
| 16 | 历史对照通过重新审计 | 0.8008724 | 0.0826776 | 0.0000000 | 0.0000000 |
| 32 | 完成并通过审计 | 0.7991998 | 0.0830104 | -0.0016726 | +0.0003328 |
| 64 | 首批 OOM，无有效结果 | — | — | — | — |

本表仅纳入最终 Test evaluation 且通过协议、完成状态、覆盖及双日志核对的指标。
当前三个有效维度中，16 维的宏平均 AUC 最高、LogLoss 最低；不将缺失的 64 维
误写为性能退化，也不从单种子结果推断普遍最优。逐标签指标位于 results/label_metrics.csv。

64 维 attempt 1 在两张 H100 80GB 上的首个编译训练前向中失败，两个 rank
均申请额外 816 MiB 时 OOM，单卡仅剩约 533 MiB。失败证据位于
notes/r64_attempt1_failure.json，原始日志及 attempt 1 的全部记录均保留。

需固定的协议：

- token_dim=256、num_heads=2、num_layers=3；原每头 Q/K 维度为 128。
- 所有六处 SISA 接入点统一使用同一个 r；接入位置和衰减参考保持原设置。
- seed=20262028，rank 使用 seed+rank；dataloader_seed=2027；
  sisa_parameter_seed=20260822，保留原有接入点种子偏移。
- 两卡 DDP、每卡 batch=16384、global batch=32768、accumulation_steps=1。
- 训练 1 epoch、max_len=100，原数据划分、预处理、标签顺序、drop_last、
  验证及测试覆盖、验证集选择 checkpoint 的规则保持一致。
- 原 AdamW/Adagrad、学习率、BF16、torch.compile 和 checkpoint 设置保持一致。
- sparse_optimizer_foreach=False、sparse_adagrad_chunk_size=16777216，
  CUDA allocator 沿用 expandable_segments:True。
- sisa_variant=full、lambda_init=0.1、score_scale=1.0，衰减和旋转均启用；
  decay_bias=-5.0、exp_clamp=11.0 沿用实现默认值。
- 历史资源为两张 H100、16 CPU、480 GiB 主机内存；记录实际硬件及软件环境。
  GPU 型号可不同，但训练和评估协议必须匹配；跨硬件运行时间不能直接解释为维度差异。

正式提交前已生成独立的冻结配置、任务映射和协议比较记录，225 项字段比较通过，
已重算当前六个 split/rank 的数据覆盖并与历史逐项对齐。允许的有效配置差异为 `sisa_score_dim`
及输出标识 `model_id`；旧日志未记录 `sisa_variant` 时按实现默认 `full` 归一比较。
提交及完成后还须核对实际日志，不能将此设计检查视为新实验完成。

## 已完成的实现检查

2026-09-23 在 CPU 上分别构建本轮四种维度的 HyFormerBlock，使用 input_dim=256、
num_heads=2、num_layers=3、num_ns_token=4、num_global_token=2。
相同骨干初始化种子下，排除 SISA 参数后的 state_dict 逐张量完全相同。
对每种维度的全部六个 SISA 模块检查了输出形状、有限值、输入与参数梯度，
以及全 padding 样本的零偏置；上述检查均通过。该检查未训练完整模型，
也未验证 GPU 显存峰值、吞吐或编译内核。

| 每头维度 r | 六处 SISA 的可训练参数总数 |
|---:|---:|
| 8 | 64,536 |
| 16 | 125,976 |
| 32 | 248,856 |
| 64 | 494,616 |

这些是当前 HyFormer 配置下的 SISA 分支参数量，不是整个模型的参数量。
单处隐藏宽度 D、头数 H 的完整 SISA 参数量为
`(5/2)×H×r×D + H×D + 2×H`。

## 结果记录与解释边界

报告各标签及等权宏平均 AUC、LogLoss，相对 16 维的差值，SISA 分支参数量及
整个模型参数量。若采集耗时、吞吐和显存，应使用相同硬件与一致测量口径，
区分编译、训练和评估开销。训练完成后统一展示全部预设维度，包括退化结果。
维度选择若用于调参，应依据验证集；测试集曲线仅用于预设实验的最终报告。

本轮衡量当前实现对 r 的整体敏感性。维度变化会改变内容投影宽度、旋转平面数、
参数量和初始化分布；当前偏置未额外乘以 `1/sqrt(r)`，因此分数尺度也可能变化。
保持当前缩放方式，不同时引入新的归一化规则。SISA 使用隔离的初始化生成器，
骨干初始参数可保持一致，但跨维度的 SISA 参数不保证为相同数值或嵌套子集；
即使衰减参数形状相同，也会受此前不同尺寸投影的随机数消耗影响。

单个种子、单个模型—数据集组合仅支持该设置下的描述性维度曲线，不能声称
统计显著、普遍最优或已将纯维度效应与容量、初始化尺度效应分离。

## 证据位置

- 实现：`unirank/pytorch/layers/attentions/sisa.py`、`model_zoo/HyFormer.py`。
- 参数入口：`run_expid.py`。
- 已有冻结协议：`experiments/ablations/sisa_components_hyformer_merrec_bcd/configs/`。
- 历史原始日志：上述 run ID 在 `checkpoints/MerRec_Action/` 下的 `.log` 文件。
- 历史集群日志：`logs/unirank-sisa-seed20262028-ws2-548166_1.out`。
- 新运行的原始日志、检查点及审计中间产物应分别放入 `logs/`、`checkpoints/`
  和 `artifacts/sisa_score_dim/`；最终紧凑结果放在本研究目录的 `results/` 下。

## 运行和守护

数组映射固定为 task 0→r8、task 1→r32、task 2→r64，最多两个任务同时运行。
实际 job/task 为 `643736_0`、`643736_1`、`643736_2`。16 维为历史 `548166_1`。
提交前 141 项 CPU 回归测试通过，两个新增 shell 脚本通过 `bash -n`。
启动器为 `scripts/run_sisa_score_dim.py`，Slurm 脚本为
`scripts/submit_sisa_score_dim.sbatch`。独立运行标识中包含维度和 attempt，
已有日志、模型或运行目录存在时拒绝覆盖。

`scripts/collect_sisa_score_dim_results.py` 核对任务身份、运行清单、源码散列、
全部有效配置、数据覆盖、维度变化后的预期参数量、最终五标签 Test 指标及
Slurm 完成状态。未完成或校验失败的记录保留状态，但不接纳其指标。

`scripts/monitor_sisa_score_dim.sh` 按用户要求使用 `sleep 600`，每十分钟
采集一次状态和进度，将快照放入 `artifacts/sisa_score_dim/monitor/`。
全部完成并通过审计时结束；全部进入终态但有失败或审计问题时以失败状态结束，
保留诊断证据。重试需记录原因并增加 attempt，不改变既定实验协议。
