# RISE 在腾讯数据集上的 Attention 重分配实验

日期：2026-09-23。模型：HiFormer；数据集：TencentGR_10M_Action；训练种子：20262028。

**实验已完成并通过核验。** 在固定抽取的512条测试交互、474位用户上，target attention的平均TV为0.164466，中位数为0.163029；401条交互的最高权重位置发生改变，占78.32%。这些数字描述同一已训练模型内部加入局部分数修正后的注意力重分配，不是预测准确率或AUC提升。

本次交付包括一张中位数案例图、一张TV经验累积分布图、两个附录案例，以及逐位置数据和复核记录。没有新增训练，没有更改骨干attention实现，没有用该诊断样本替代正式测试评估。

## 1. 图中展示什么

观测站点是 `attention_layers.dot_attention`：query对应候选物品，keys对应历史物品及其记录动作。该target attention实际只有一个头，固定展示head 0。HiFormer后续字段交互层的头数不适用于这里。

从compiled BF16前向中捕获实际Q、K、最终Delta和真实mask，在FP32中重建以下分布：

\[
S=QK^\top/\sqrt{d},\qquad a^{(0)}=\operatorname{softmax}(S),\qquad
a=\operatorname{softmax}(S+\Delta).
\]

这里d=256，原score除数为16。Delta按原前向规则转换为Q的数据类型后再进入FP32重建；所有softmax都在完整允许键集合上计算，保留原padding mask。图中的权重是实际捕获张量的数学重建，不声称是融合attention kernel显式返回的权重。

| 图中行 | 量 | 阅读方式 |
|---|---|---|
| 修正前 | a⁽⁰⁾ | 同一RISE模型在该局部位置不加Delta时的权重 |
| 分数修正 | Delta | RISE实际加入的有符号分数修正 |
| 修正后 | a | 加入Delta后的权重 |
| 权重变化 | a−a⁽⁰⁾ | 注意力实际增加或减少的量 |

修正前后共用色标；分数修正使用以零为中心的蓝橙色标；权重差使用有零线的正负柱。三个案例也共用注意力与Delta色标，便于横向比较。横轴保留全部100个原始历史位置，灰色区域是padding，不进行截断后的重新归一化。位置表示输入序列顺序，不表示真实时间间隔。

## 2. 固定样本上的变化幅度

每个测试物理part取正式保留区间内的首16384行窗口，用seed 20260923及part_id、row_offset的SHA256优先级抽64行；八个part共512行。采样没有读取标签、预测收益或attention变化。实际模型仍以完整16384行执行前向，前向后才提取指定64行。

该采样框是固定窗口，不是全测试集均匀代表样本。512条样本均有非空历史，历史有效长度为26–99；没有用于说明空历史行为的实测样本。每条交互在汇总中等权，同一用户可能出现多次。

\[
\operatorname{TV}(a,a^{(0)})=\frac12\sum_j|a_j-a_j^{(0)}|.
\]

| 指标 | 结果 |
|---|---:|
| 测试交互数 | 512 |
| 不同用户数 | 474 |
| 非空历史数 | 512 |
| 空历史数 | 0 |
| 平均TV | 0.164466 |
| TV中位数 | 0.163029 |
| TV第10百分位 | 0.118213 |
| TV第90百分位 | 0.210034 |
| 最高权重位置改变数 | 401 / 512 |
| 最高权重位置改变比例 | 78.3203% |

平均TV可理解为平均约16.45%的注意力概率质量重新分配。最高权重位置的改变是另一个指标：首位间距很小时，较小的权重变化也可能改变首位。两者均不能推导预测性能改善。

分布图采用ECDF，纵轴是在这些非空诊断样本中TV不高于横轴阈值的交互比例，并标出三个选定案例。

## 3. 中位数案例：具体从哪些位置转移到哪些位置

案例选择规则在读取attention结果前固定：选TV最接近第50百分位的样本；等距时按part_id、row_offset排序。附录同样选第10和90百分位附近样本，不按预测是否改善挑选。

| 案例 | sample_index | part_id | row_offset | 有效历史长度 | 实际TV |
|---|---:|---:|---:|---:|---:|
| 正文：接近P50 | 128 | 2 | 413 | 92 | 0.162977 |
| 附录：接近P10 | 429 | 6 | 12230 | 90 | 0.118193 |
| 附录：接近P90 | 150 | 2 | 5025 | 70 | 0.210062 |

正文案例的候选物品编码为5126221。下面列出该案例内权重绝对变化最大的五个历史位置，按原位置排列。此处只是对已选案例标注重点位置，不改变样本选择，也不删除其余历史位置。

| 历史位置（从1计） | 历史物品编码 | 已记录历史动作 | Delta | 修正前权重 | 修正后权重 | 权重差 |
|---|---:|---|---:|---:|---:|---:|
| 23 | 4295713 | click | +0.129883 | 0.037587 | 0.023012 | −0.014575 |
| 97 | 14390345 | exposure | +1.507812 | 0.012163 | 0.029538 | +0.017375 |
| 98 | 15854583 | exposure | +1.718750 | 0.010025 | 0.030062 | +0.020038 |
| 99 | 5703651 | exposure | +1.781250 | 0.011641 | 0.037160 | +0.025519 |
| 100 | 99146 | exposure | +2.000000 | 0.010770 | 0.042787 | +0.032017 |

这张图直接展示：在该案例中，部分权重从较前的历史位置转向序列末端的若干曝光记录。例如位置23的Delta虽然为正，其最终权重仍从3.7587%下降至2.3012%；位置100则从1.0770%增加至4.2787%。原因是softmax取决于所有允许位置修正后的相对分数，单个Delta的正负不能独立决定权重升降。

历史动作名称来自数据metadata确认的映射：click=1、conversion=2、exposure=3。物品编码保持匿名，不编造商品类别。候选物品的action=0是输入中的刻意屏蔽值，不能解释成其真实动作。该案例不证明末端曝光记录更有价值，也不恢复用户真实兴趣。

## 4. 可直接用于论文的图注和正文

建议放在组件消融之后，作为“Attention Redistribution”小节；方法部分在score correction公式附近指向该实测案例。

**中文图注：** 同一HiFormer–TencentGR RISE检查点中的target-attention重分配。固定候选物品query、head 0、原生分数与允许历史键，依次展示局部分数修正前权重、实际Delta、修正后权重和权重差。正文案例按固定诊断样本TV中位数邻近规则选择，未按预测收益筛选。修正前后共用色标，灰色为padding，所有权重均在完整允许键集合上归一化。权重由实际compiled Q/K/Delta进行FP32重建；“修正前”不是独立训练的Baseline。

**English caption:** Target-attention redistribution within one trained HiFormer–TencentGR RISE checkpoint. For the same candidate query and head 0, the aligned rows show the pre-correction weights, captured score correction, post-correction weights, and their difference. The example is selected by proximity to the median TV in the fixed diagnostic sample, without prediction-based selection. Before and after share a color scale; gray positions are padding. We reconstruct softmax weights in FP32 from captured compiled tensors over the complete allowed-key set. The pre-correction reference is not an independently trained baseline.

**English results paragraph:** We examined target attention in a fixed diagnostic sample of 512 test interactions from 474 users. Instrumented and uninstrumented predictions were bitwise identical for both action labels and also matched the corresponding historical evaluation predictions. The mean total variation distance between locally uncorrected and corrected attention distributions was 0.1645, with a median of 0.1630. The highest-weight history position changed in 401 of 512 interactions (78.32%). The median-TV example illustrates that a positive score correction need not increase the corresponding attention weight, because normalization depends on the corrections at all allowed positions. These observations describe within-checkpoint redistribution on fixed test windows and do not establish a causal explanation of ranking gains or population-level generality.

## 5. 数值验证与运行记录

使用原有full RISE检查点，训练job/task为548166_14。训练协议为seed 20262028、2张H100、每卡16384、global batch32768、1 epoch、history上限100、BF16和torch.compile。此次为单GPU诊断推理，每个前向仍保留16384行输入形状，共八个完整窗口；不是新的完整测试评估。

检查点SHA256：`a4a34eb4e0cb0b17f961752b583a69b0cf2d5c86019d15381d2a9ecae2206ca5`。

| 核验 | 结果 |
|---|---|
| 成功作业 | 646303，ACD1-63，COMPLETED，退出码0:0 |
| 成功作业耗时 | 00:01:36 |
| 采集前/后，两个标签、512行 | 逐位相等；最大绝对误差0 |
| 本次/历史正式预测，两个标签、512行 | 逐位相等；最大绝对误差0 |
| 采样行号、标签、用户、物品、历史长度 | 与历史预测按物理行身份一致 |
| 源码与产物哈希 | 独立核验61个源码、11个运行产物 |
| 独立float64 softmax重建最大绝对差 | 修正前8.68×10⁻⁹；修正后1.17×10⁻⁸ |
| 归一化、padding质量、TV上界、首位索引 | 全部通过 |
| 最终完整CPU回归 | 170项通过 |
| 专项CPU回归 | 16项通过，含真实torch.compile缓存/hook回归 |
| Slurm脚本语法 | bash -n通过 |

失败尝试完整保留：645949使用64行微批，尚未采集attention时即未通过历史预测门槛，两个标签最大绝对差均为0.001953125；646265恢复完整batch后与历史预测逐位一致，但缓存编译图跳过新hook，捕获数为0，因此整项拒绝采用。646303在注册hook后重置Dynamo缓存，最终通过全部门槛。没有降低一致性要求，没有更换512条样本，没有从失败尝试中采用机制数值。

历史数据内容并未由该实验追溯证明与原训练时逐字节一致；此次冻结并复核的是当前输入文件身份、metadata及源码，并通过历史逐行预测比对确认所观测样本的实际一致性。

## 6. 交付与复现

正文使用 `figures/attention_main_p50.pdf` 和 `figures/attention_tv_ecdf.pdf`；附录使用 `figures/attention_appendix_p10.pdf`、`figures/attention_appendix_p90.pdf`。每张图均有同名SVG、PNG。

`figures/selected_case_positions.csv` 包含三个案例全部100个位置的绘图数值；`figures/case_selection_receipt.json` 保存样本选择、色标范围及输入哈希；`results/` 保存512行compact统计、预测核验、来源manifest和独立审计；`audit/` 保存失败记录、测试及视觉QA。

原始权重、完整输入batch缓存和逐位置数组保留在 `artifacts/rise_attention_redistribution/tencent_646303/`，完整Slurm日志在 `logs/unirank-rise-attention-tencent-646303.out`。不把这些大体积运行文件纳入正式报告目录。

复现命令均从仓库根目录执行：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_rise_attention_redistribution.py' -v
bash -n scripts/submit_rise_attention_redistribution.sbatch
# 仅在确有重新采集需求时提交；已完成作业无需重跑。
sbatch scripts/submit_rise_attention_redistribution.sbatch
.venv/bin/python scripts/audit_rise_attention_redistribution.py --input-dir artifacts/rise_attention_redistribution/tencent_646303 --output-dir experiments/rise_attention_redistribution/results --job-id 646303
experiments/rise_sample_analysis/figure_env/.venv/bin/python scripts/plot_rise_attention_redistribution.py --input-dir artifacts/rise_attention_redistribution/tencent_646303 --output-dir experiments/rise_attention_redistribution/figures
```

图形环境沿用已有uv锁定环境。实验只支持这个检查点、站点与固定采样框下的重分配观测；没有给出AM/PA独立因果贡献、训练种子稳定性或其他模型数据集上的共同机制结论。
