# HyFormer–MerRec 固定表示机制观测与稳态推理基准

日期：2026-09-23。范围仅为已经训练完成的组件参照 B（content-only）与 E（完整 RISE）；如需原生 A 的效率列，必须提供独立训练、协议匹配的 A checkpoint log。**B 不是原生 Baseline。** 本轮新增代码不改变 `export_predictions.py`、controls、core 模型、训练配置或 score_dim 实验。

## 1. 观测问题与解释单位

论文中的机制命题是在同一模型、同一插入点、同一输入表示和同一原生 Q/K/V 下讨论的。因而不能将两个分别训练好的模型 attention 差异全部归因于一个固定的修正 Δ。本脚本在 B 或 E 的一次前向中捕获实际 Q、K 和传入 attention 的实际 Δ，比较该位置的

\[
p_j=\operatorname{softmax}_j(s_{ij}),\quad
\widetilde p_j=\operatorname{softmax}_j(s_{ij}+\Delta_{ij}),
\quad s_{ij}=q_i^\top k_j/\sqrt{d_h}.
\]

这里“未加 Δ”是对当前模型当前位置的条件反事实，并不是独立训练的 native backbone 性能。移除 Δ 后不向下游继续前向，不声称这等于模型消融后的 AUC。

观测使用实际 low-precision Q/K 和实际 score_bias，在关闭 autocast 的诊断算子里以 FP32 计算 logits/softmax。仓库使用 fused SDPA，内部概率不能通过常规 forward hook 直接拿到，因此本结果应称 **重构的 pre-dropout softmax selection**，不可称 SDPA kernel 实际导出的 attention weights。重构不会替换模型输出。

## 2. 六个站点和 padding 语义

每层有 sequence self-attention 和 global-query-to-history attention，各三层，总计六处。观测会验证站点数量；缺少站点或重复样本/站点记录直接失败。

当前 HyFormer 在这些 softmax 站点没有传入 padding hard mask，而是在输入和输出等位置将序列 token 乘 mask。RISE 的无效辅助投影为零，但 native 的注意力仍允许指向 padding keys。为保留原语义：

- 所有 native 允许的 keys，包括 padding，均参加 p、修正后 p、TV、Ω 和 top-key 比较。
- sequence self-attention 的无效 query 不进入 query 平均；query decoding 的 global queries 均有效。
- 同时报告原生/修正后的 padding probability mass。
- Δ 的正负比例既报告所有允许 key 的版本，也报告只保留有效 history keys 的版本，避免短历史更多零修正 padding 造成机械差异。
- AM/PA 的成对变换统计限定有效 query/key 端点。

不能先给原生注意力加一个新 mask，再把结果解释成当前 checkpoint 的机制。

## 3. 输出的观测量

每行对应一个 variant、source row、attention site；head/query 先在该样本内平均。站点汇总为等样本平均，不把长历史的更多 pair 自动赋予更高权重。

| 类别 | 观测量 | 用途与边界 |
|---|---|---|
| 修正尺度 | `omega_mean/max`；Δ均值、绝对值、正负比例；centered Δ/native logit SD | 判断修正是否足以改变选择；lambda 单独大小不能代替它 |
| 概率再分配 | TV mean/max、`tanh(Ω/4)`、最大 bound violation | 检查论文条件性上界；不是预测 AUC 增益的公式 |
| 选择排序 | top-key changed fraction；native top gap；`gap>Ω` 的充分保持条件及其违例比例 | 区分“有非零修正”与“实际改变首选项” |
| Padding | native/corrected padding mass | 检查收益或变化是否伴随对 padding 的再分配 |
| AM | 成对 log-amplitude、amplitude、放大比例、局部log-decay均值 | 量化调制强度；根据 site reference 解释，不能一概称统一时间衰减 |
| PA | phase increment/prefix绝对值、pair cosine变化、alignment符号变化比例 | 量化当前投影的方向变化，不把大相位自动解释成有效匹配 |
| 数值稳定性 | query/key clipping比例 | 判断未clipping理论解释覆盖多少实际情况 |
| 固定表示代数去除 | `frozen_am_delta_effect_abs_mean`、`frozen_pa_delta_effect_abs_mean` | 在当前表示/权重下重算单项变换的代数变化；不等于重新训练组件的收益 |

AM/PA 代数诊断使用其实际控制投影与 deterministic scan；中间统计在 FP32 重构，可能与最终 BF16 channel round-off 有极小差异。核心 TV/Ω 指标直接采用实际传入 attention 的 Δ。全空历史样本在 self-attention 的有效 query 数为零，其平均指标保留 NaN，不伪填 0；global-query 站点仍可观察 native padding mass。

## 4. 样本固定和存储

默认从历史两 rank 各前 2 个原始批次中，以 `(seed, part_id, row_offset)` 的 SHA256 优先级抽取每批 64 行，默认 seed20260923。采样不读取标签或预测。历史 loader 的 world-size2、per-block drop_last 和原始 row offset 不变；抽取后用小批次做单卡推理。

这是预先声明的 loader window 内随机优先级样本，**不是全测试集的均匀代表样本**。它可用于实现核查、机制案例与局部统计；若用于论文分组总体结论，需要进一步扩展覆盖不同 parts/用户分布并明示采样权重。脚本写出采样frame和完整 row IDs，并保存小规模 `sample_batches.pt` 以让所有后续变体和benchmark复用完全相同输入。

不保存完整 attention、Q/K/V 或 Δ矩阵。输出为逐样本站点标量、预测校对表、输入小批缓存、来源与核验 JSON。原始大型 checkpoint 只读；全路径位于 `artifacts/rise_sample_analysis/mechanism/<run_id>/`。

## 5. compiled–eager 一致性是解释前置门槛

先通过原 `export_predictions.load_model` 加载带 `torch.compile` wrapper 的正式模型，`strict=True` 读 checkpoint，获得相同小批的 compiled 预测。随后销毁模型、释放 GPU cache，构造关闭compile的观测模型。

Eager映射只允许移除完整路径段 `_orig_mod`；映射后的全部 key、shape、dtype 必须与 eager state_dict 完全一致。别名碰撞、缺失键、额外键、shape或dtype不一致均失败，不使用 `strict=False`。

在 eager 模型上注册只读 hook，一次前向同时获得机制统计和所有5标签预测，与 compiled 预测逐行比对。默认门槛事前固定为

\[
|p_{eager}-p_{compiled}|\le 5\times10^{-4}+5\times10^{-3}|p_{compiled}|.
\]

保存每标签最大/平均绝对差、失败行数与阈值。这是数值兼容性检查，不是 statistical equivalence test。若任何标签失败，统计文件保留 `_unvalidated` 后缀，脚本退出失败，不产生 `complete.json`；不能先调整阈值直到通过再不披露。

可选 `--prediction-dir` 进一步校对既有正式导出的 parquet：必须同 checkpoint SHA256、compiled BF16，并逐行检查 part/offset、用户/物品、有效长度和标签。它额外报告 compiled microbatch 与原始大batch export 的差异，可区分 compile差异与batch shape引起的数值差异。该步骤需要每个被抽样原始 shard 都已导出。

## 6. 推荐先跑的 GPU smoke

该程序是现有模型推理观测，用一张GPU；不能将其称作单卡正式训练结果。

```bash
.venv/bin/python experiments/rise_sample_analysis/scripts/observe_mechanism.py \
  --variants E \
  --batches-per-rank 1 --samples-per-batch 16 \
  --output-dir artifacts/rise_sample_analysis/mechanism/smoke_YYYYMMDD_jobid
```

第一次成功后可使用更大的B/E同一采样研究：

```bash
.venv/bin/python experiments/rise_sample_analysis/scripts/observe_mechanism.py \
  --variants B E \
  --batches-per-rank 4 --samples-per-batch 64 \
  --output-dir artifacts/rise_sample_analysis/mechanism/paired_YYYYMMDD_jobid
```

复用已审计小批可以加 `--sample-cache artifacts/rise_sample_analysis/mechanism/<previous>/sample_batches.pt`。该模式会保留cache来源与新cache哈希；不得据此宣称扩大了采样覆盖。

每个variant接受状态需要：`<variant>_prediction_gate.json` 的passed=True、`<variant>_observations.parquet`（无unvalidated后缀）、六站点完整且row IDs唯一，最终 `complete.json`。`<variant>_site_means.csv` 是描述性均值，不能替代逐用户不确定性估计。

## 7. 独立的效率脚本

`scripts/benchmark_inference.py` 对 B/E 采用同一GPU、同一缓存batch、同一precision和同一mode；每个variant在新的子进程中构建模型。输入张量先搬到GPU，计时范围仅 `model.forward`，不含 parquet读取、collator、H2D、checkpoint读取或初次compile。默认compiled，eager必须另跑一份独立结果。

```bash
.venv/bin/python experiments/rise_sample_analysis/scripts/benchmark_inference.py \
  --sample-cache artifacts/rise_sample_analysis/mechanism/<observation>/sample_batches.pt \
  --mode compiled --warmup 30 --blocks 10 --repeats 100 \
  --output-dir artifacts/rise_sample_analysis/mechanism/<benchmark>
```

程序强制 `eval()`、`no_grad()`、原BF16/autocast，第一次forward单独记录并排除。warm-up之后在CUDA event block边界同步；每block100次forward取平均，报告10个block平均值的median、Q25/Q75。它们不是请求级tail latency，也不是跨训练seed置信区间。样本batch默认来自cache第一项；改变batch大小需要生成对应cache，不能直接拿不同batch的倍率比较。

显存报告warm-up后的allocated/reserved baseline、测量窗内peak allocated/peak reserved以及peak allocated减baseline。reserved不是活跃tensor内存，不能把两者混写。各variant fresh subprocess避免另一模型/compiled graph残留；checkpoint、sample cache、脚本/core源码均保存哈希。

计时后另外执行一次 profiler forward，记录attention/Flash/softmax/Triton有关event名称；不把profiler开销计入延迟或peak memory。记录GPU UUID/型号、torch/CUDA、节点、可见设备、功耗/时钟和compute-process列表。应在独占或确认无并发工作的GPU上运行；其他进程列表本身需要人工检查，脚本不把“同一块GPU”自动当成“无并发干扰”。

每个独立round可指定 `--variant-order B E` 或 `--variant-order E B`，并用 `--round-id forward01`/`reverse01` 和不同 output-dir 保存；默认 B E。顺序必须恰为实际模型集合的排列；存在native log时才能指定包含A的排列。建议在同一设备用相同cache做正反顺序至少两轮，并审阅温度/功耗/时钟与并发状态。不要将一次固定顺序的结果包装为高精度的普适硬件结论。

可选加 `--native-log checkpoints/MerRec_Action/<matched_A>.log` 才启用A。程序检验其非SISA协议与E匹配；不会把B改称Baseline，也不会用E权重临时关闭模块伪装训练过的A。输出只有 E/Bcontent-only 和存在A时的E/A两种倍率，并保留各自绝对延迟和memory。

## 8. CPU检验与尚未验证事项

`tests/test_mechanism.py` 的CPU tests覆盖：两key TV上界等号例；常数修正不改变selection；hard mask不能被大Δ重新打开；保留padding mass；排除无效query；B的AM/PA恒等；极端E检测clipping；真实六站点HyFormerBlock安装hook前后逐tensor bitwise一致；严格checkpoint键映射；固定采样与cache；数值gate拒绝超差/NaN；block timing统计不冒充tail latency；benchmark顺序必须精确覆盖实际可用模型。最初10项已通过，增加顺序校验后合计11项。

```bash
.venv/bin/python -m unittest discover -s experiments/rise_sample_analysis/tests -p test_mechanism.py -v
```

这些测试不加载实际3.24GB权重，不调用GPU编译，不证明actual checkpoint的数值gate会通过。真实权重加载、BF16/compiled兼容、GPU kernel路径、latency和memory仍需GPU脚本成功执行后才能报告。本子任务只实现与CPU检验，未提交或占用GPU。
