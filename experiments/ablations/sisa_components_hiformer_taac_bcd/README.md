# SISA 组件消融：HiFormer–TAAC-25

新增第三个模型—数据集组合：HiFormer–TencentGR_10M_Action（TAAC-25）。
只训练 B/C/D 三个任务，一个 seed，完整 SISA E 复用 H100 seed 20262028 的既有结果。
原有 RankMixer–KuaiRand 和 HyFormer–MerRec 两组任务保留，不重复提交。

## 选择依据与研究问题

标准主实验 seed 20262027 中，该组合任务宏平均 AUC 从 0.855235000
提升至 0.866796000，增量为 0.011561000；两个任务 AUC 均提升。
宏平均 LogLoss 从 0.032188500 降至 0.027953500。
新增组补充另一种模型结构及广告点击、转化任务场景，用于检查组件贡献。
三组均根据已知主实验表现选取，不能据此推断全部 16 个标准组合的组件贡献。

本轮直接对照事先固定为 H100 首轮 seed 20262028，宏平均 AUC 为
0.870858000，宏平均 LogLoss 为 0.030295000。旧四卡 baseline 不作为严格配对对照。
不根据新增消融结果更换对照种子，不新增 A/E 训练。

## 新增任务及唯一变量

| 数组任务 ID | 组别 | 配置值 | 消融含义 |
|---|---|---|---|
| 0 | B | content_only | 去掉衰减和旋转，保留基础内容投影 |
| 1 | C | decay_only | 去掉旋转，保留内容投影与衰减 |
| 2 | D | rotation_only | 去掉衰减，保留内容投影与旋转 |

三个任务均从头训练，保持原完整 SISA 的接入位置、内容投影、每头可学习缩放、
骨干与已有掩码处理。关闭衰减为 log(alpha)=0，关闭旋转为相位为零。
共有参数的隔离 RNG 初始化顺序保持一致，未启用分支冻结且不参与前向计算；
冻结参数不计为有效容量匹配。
HiFormer 包含目标注意力和特征融合 token 注意力接入位置，融合 token 不等同于
时间事件，不能将所有位置的收益统一解释为历史遗忘或时间衰减。

## 固定 H100 对照协议

- 实验 seed 20262028；各 rank 使用 seed+rank。
- 数据加载 seed 2027；SISA 参数 seed 20260822，接入位置偏移沿用原实现。
- 2×H100，每卡 batch 16384，全局 batch 32768，梯度累积 1。
- 训练 1 epoch，最大序列长度 100，BF16 与 torch.compile 启用。
- SISA score_dim=16、lambda_init=0.1、score_scale=1.0，其余默认设置保持不变。
- CPU 16 核、主机内存 480 GiB；数据预处理、特征、标签顺序、优化器、学习率、
  数据分块、drop_last、验证和测试流程均匹配本组合对应的 H100 完整 SISA 轮次。
- 不应用 MerRec 专用优化器覆盖或 allocator 设置。

configs/base_configs.json 冻结当前基础 YAML，reference_params.json 保存历史原始日志
全部参数。configs/protocol_comparison.csv 对比三组最终有效设置，允许差异仅为
sisa_variant 及用于区分输出的 model_id；其他配置差异在启动前报错。
configs/data_coverage.csv 重算并对比当前两 rank 的训练、验证、测试分配。
行数和块数一致不能替代历史数据文件完整散列验证，正式完成后还需检查运行日志覆盖。
对照身份、原始任务指标和审计边界详见 notes/reference_audit.md。

## 评估与归档

报告 is_click、is_conversion 两个任务的 AUC、LogLoss，以及等权任务宏平均。
比较 C−B、D−B、E−C、E−D；AUC 描述性交互量为 E−C−D+B。
单种子结果不报告跨种子标准差或显著性，不省略退化项。
没有固定位置规则对照，本轮不能独立证明内容自适应优于固定规则。

提交脚本：scripts/submit_sisa_components_hiformer_taac_bcd.sbatch。
共享启动器：scripts/run_sisa_component_ablation.py --study sisa_components_hiformer_taac_bcd。
任务映射为 configs/tasks.csv，提交和验证记录在 notes/launch.md。
数组固定 0–2，最多两个任务并行。重试增加 attempt，禁止覆盖已有日志或模型。
原始输出保存于 logs/、checkpoints/、artifacts/sisa_components_hiformer_taac_bcd/，均由 Git 忽略。
