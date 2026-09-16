# SISA 组件消融：RankMixer–KuaiRand

本轮只新增一个模型—数据集组合的 B、C、D 三个训练任务。
模型为 RankMixer，数据集为 KuaiRand_Video_Action；不扩展到其他组合或随机种子。
E（完整 SISA）复用已归档的 H100 首轮复跑，seed 为 20262028。
原始四卡 baseline A 不作为本轮的严格配对对照，不新增 A 或 E 的训练。

## 选择依据

在当前标准 4×4 主实验范围内，该组合完整 SISA 的任务宏平均 AUC 从
0.852376 提升至 0.860973167，绝对增量为 0.008597167；宏平均 LogLoss 从
0.179610167 降至 0.174766833。六个任务的 AUC 均提升。
后续两个 SISA 轮次的宏平均 AUC 分别为 0.861501500 和 0.864025833，
本轮事先指定其中首轮（seed 20262028）作为直接对照，严格匹配其两卡协议，
不根据后续消融结果更换对照种子。
RankMixer 的 SISA 接入目标注意力汇聚位置，适合拆分组件贡献。

该组合根据已知主实验收益选取，用于分析方法有效场景下的组件贡献。
结果不能代表全部 16 个组合，也不据此推断未实验场景的普遍有效性。

## 三个新增任务

| 数组任务 ID | 组别 | 配置值 | 内容打分 | 动态衰减 | 动态旋转 |
|---|---|---|---|---|---|
| 0 | B | content_only | 有 | 无 | 无 |
| 1 | C | decay_only | 有 | 有 | 无 |
| 2 | D | rotation_only | 有 | 无 | 有 |

三组均从头训练。关闭衰减时 log(alpha)=0，关闭旋转时相位为零。
所有分支仍按历史顺序使用隔离 RNG 初始化，从而保持共有参数的初值一致；
未启用分支被冻结且不参与前向计算，不能把保留的冻结参数称为有效容量匹配。
每头可学习缩放、内容投影、原始 Q/K/V、掩码及骨干结构均保留。

## 固定训练与评估协议

- 实验 seed：20262028；每个 rank 使用 seed+rank。
- 数据加载 seed：2027；SISA 参数 seed：20260822，接入位置偏移规则保持不变。
- 2 GPU DDP，每卡 batch 16384，全局 batch 32768，梯度累积 1。
- 训练 1 epoch，最大序列长度 100，BF16 与 torch.compile 保持启用。
- SISA score_dim=16、lambda_init=0.1、score_scale=1.0；其余默认值保持主实验设置。
- 数据划分、预处理、任务标签、优化器、学习率、drop_last 和评估流程沿用主实验。
- configs/base_configs.json 冻结基础 YAML 配置；运行时覆盖为上述 H100 轮次设置。
- configs/reference_params.json 保存原始 H100 日志的全部 72 个配置字段。
  configs/protocol_comparison.csv 逐项对比三组的最终有效配置，允许差异仅为
  sisa_variant 和用于区分输出文件的 model_id；其他差异会在启动前阻止任务。
- 本轮与对照均为 acd_u 两张 H100，CPU 16 核、主机内存 480 GiB。
  最终结果须核对逐 rank 数据覆盖与评估样本数，不能仅检查全局 batch。
- 单种子结果不报告跨种子标准差或统计显著性。

## 复用对照

| 组别 | 历史 job/task | 原始 run ID |
|---|---|---|
| E | 548166_7 | SISA_single_seed_RankMixer_KuaiRand_Video_Action_sisa_seed20262028_ws2_bs16384_acc1_attempt1 |

E 的六任务宏平均 AUC 为 0.861501500，宏平均 LogLoss 为 0.174569667。

对照来源：experiments/sisa_paper_report/archive/results/cell_results.csv、
label_results.csv、run_evidence.csv、logged_configurations.json。
正式训练完成前不填写 B/C/D 的指标。

## 结果解释

主指标为六个任务的等权宏平均 AUC，辅助指标为宏平均 LogLoss；同时报告全部任务。
C−B 衡量加入衰减的收益，D−B 衡量加入旋转的收益；E−C、E−D 衡量组合中的增量贡献。
E−C−D+B 是 AUC 的描述性交互量，单次结果不支持稳定协同或显著性声明。
如果 B 与 E 相近，应如实说明组件必要性尚未得到支持。
本轮不包含固定规则衰减/旋转对照，因此不能单凭本轮声称内容自适应优于固定位置规则。

## 启动、归档与检查

提交脚本：scripts/submit_sisa_components_bcd.sbatch，固定数组 0–2，最多两个任务并行。
启动器：scripts/run_sisa_component_ablation.py；--dry-run 只检查配置并打印命令。
configs/tasks.csv 固定任务映射。重复运行必须增加 attempt，禁止覆盖已有日志和模型。
原始输出归档在 logs/、checkpoints/、artifacts/sisa_components_bcd/，均不纳入 Git。
每个任务保存源代码 SHA-256、Git 提交、实际 GPU、种子、命令及退出码。
提交后将实际 job/task 映射与验证结果记入 notes/launch.md。
