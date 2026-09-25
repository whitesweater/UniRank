# RISE attention 重分配诊断协议

2026-09-23。本协议在读取本研究的 attention 结果前固定。

范围仅为 HiFormer–TencentGR_10M_Action，既有 full RISE、seed 20262028。
不新增训练，不改变正式20单元矩阵、原始评估覆盖或模型实现。
检查点由 experiments/sisa_single_seed20262028/results/runs.csv 中唯一完成记录确定。
记录当前分支、commit、工作区状态、源码散列、原始训练参数、权重SHA256与数据文件身份。

## 采样与比较

- 8个测试物理part，每个取正式评估保留区域的首个16384行窗口。
- 对 seed=20260923:part_id:row_offset 的SHA256升序选64行，再恢复物理行序，共512条。
- 不使用标签、模型预测、损失变化、attention变化进行入样选择。
- 复用生产BlockedBatchCollator，history长度100，pre padding，候选物品不在历史中。
- 这是固定窗口诊断，不是全测试集均匀样本，不重新报告正式AUC或置信区间。
- 唯一观测站点 attention_layers.dot_attention，query为候选物品，head固定0；该target attention原实现为单头。
- 在原compiled BF16 eval前向中捕获Q/K/最终score_bias与实际mask；不重算AM/PA控制状态。
- 实际前向保留正式评估batch=16384，对每个part的完整首窗口执行；前向完成后才提取固定64行的预测和捕获张量，共观测512行。
- 采集前与采集后的相同512行、相同16384行batch、两个预测标签必须逐位相等；失败即拒绝结果。
- 将未采集的诊断预测与历史正式E导出逐行对齐；另记录跨batch数值差异，预设门槛 atol=5e-4、rtol=5e-3。它与严格采集一致性门槛分开。

## 重建和选择

按原scale，在FP32中重建 S=QK^T/sqrt(d)，实际最终Delta先按前向规则转换到Q dtype。
保留实际mask，在完整允许键集合上计算 a0=softmax(S)、a1=softmax(S+Delta)。
这些是实际捕获张量的FP32数学重建，不声称是融合kernel显式返回的权重。
空历史不定义概率分布/TV，保留身份并单列，不能以零混入统计。
非空样本检查归一化、padding质量为零、有限性和TV<=tanh((maxDelta-minDelta)/4)+1e-6。

主案例固定选择head0 TV距有效样本第50百分位最近的行；附录取第10和90百分位最近行。
距离相同按part_id、row_offset排序。先固定站点/头/分位数再读结果；不按预测收益筛选。
若样本TV相同，分位数案例允许重复，不偷偷更换选择规则。

## 图表契约

论文PDF/SVG/PNG；白底、蓝色权重、蓝橙零中心修正/差值、灰色padding。
四行对齐：a0、Delta、a1、a1-a0；a0/a1共用同一色标。
保留完整位置轴；历史内容注释仅采用真实全局物品ID、元数据确认的动作码/名称。
匿名属性不可编造语义，target action=0不可标为真实行为。
在固定窗口TV分布中标记所选案例。标题中明确模型、数据、站点、头；图注明同一checkpoint局部比较。
附录完整保留低/高变化案例和逐位置CSV。

## 证据边界

修正前不等于独立训练Baseline。TV大不代表预测提升；attention变化不是标签预测因果贡献。
本实验不声称AM/PA独立贡献、全测试总体规律或其他模型/数据集共同机制。
大体积输入、权重、原始观测与日志进入artifacts；报告与compact数据进入本目录。
输出后独立复算raw数组与摘要、检查Slurm退出和图形渲染。

## 执行修订与失败保留

首轮645949在64行微批推理与历史正式预测比较时失败，两标签最大绝对差均为0.001953125。
失败发生在attention捕获之前，无attention结果被接受。原失败artifact、日志和门槛完整保留。
后续恢复正式batch16384前向再选行；512条样本身份、选图规则、预测门槛均不变。

646265恢复完整batch后，512条两标签预测与历史正式预测逐位相等；但新注册hook被缓存编译图跳过，capture数量为0，整项任务拒绝采用。
后续按既有compiled诊断流程在注册hook后重置Dynamo缓存，再编译采集路径；仍需capture/plain逐位相等。
