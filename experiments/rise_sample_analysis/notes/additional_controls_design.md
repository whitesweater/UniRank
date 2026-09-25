# RISE 附加机制对照与效率测量：冻结设计和实现记录

本记录日期：2026-09-23。RISE 是论文名称，仓库的 `SISA` 为同一方法的实现名称。本文件独立于低维向量维数消融；所有新增对照固定 `score_dim=16`，不修改通用训练、模型或数据加载代码。以下先区分已经核查的事实、已经实现和检验的代码、尚未训练的实验，再给出论文可以接受的解释边界。

## 1. 结论与任务边界

**推荐新增两个从头训练任务，均为 RankMixer–KuaiRand，实验种子 20262028：L 为严格 token-local AM/PA，F 为输入无关、可学习的全局控制增量。** L 与完整 RISE 的活跃参数完全一致，同时承担 local-only 和 capacity-matched 两个控制职责；无需再扩展向量宽度，也无需为凑对照数量额外增加一个 capacity-only 任务。F 保留累积操作，但其控制不读取样本表示，检验简单的相对位置变换能解释多少收益。

选择此场景主要基于可识别性：RankMixer 只有历史序列的 target-attention 一个 RISE 插入点，使用 query reference；它不存在 HyFormer 序列 self-attention 的 sequence-end 参考混杂。该场景的同协议 B/C/D/E 均已完成，四份权重也都保留。完整 E 不是根据新增对照的表现重新挑选的。

新增代码已经完成 CPU 语义检验和两个任务的配置 dry-run。本子任务没有提交 Slurm，没有进行训练，没有产生新的性能结论。完整 E 在同环境下重跑可作为后续可选验证，本轮不自动增加第三个任务。

## 2. 论文与当前代码实际有哪些证据

论文 `document.pdf` 的主文 4.2、4.3，附录 A.5、A.6 明确承认 parameter-matched、local-only、fixed-control、controlled-latency 比较尚缺失。这些限制与代码一致，不应仅靠重新解读既有 B/C/D/E 表格删除。

| 已有设置 | 内容分支 | AM | PA | 是否累积 | 可以回答 | 不能单独回答 |
|---|---|---|---|---|---|---|
| A：native backbone | 无 | 无 | 无 | 无 | 方法整体与 native 的差异 | 控制机制归因；A 没有在该 20262028 组件协议重跑 |
| B：content_only | 有 | 关闭并冻结 | 关闭并冻结 | 控制退化为零 | 辅助内容打分作为参照 | 额外非线性容量与累积的区别 |
| C：decay_only | 有 | 有 | 关闭并冻结 | AM 累积 | 在当前架构中的 AM 增量 | AM 的 local 与 cumulative 区别 |
| D：rotation_only | 有 | 关闭并冻结 | 有 | PA 累积 | 在当前架构中的 PA 增量 | PA 的 local 与 cumulative 区别 |
| E：full | 有 | 有 | 有 | 二者累积 | 完整组合的效果 | 增参、局部非线性、输入适应性等解释的排除 |

每个插入点的全参数数量为

\[
P_E=2HrD+HD+H+H(r/2)D+H
    =H[(5r/2+1)D+2].
\]

这里 `D` 是 RISE 收到的 hidden 维数，不总等于 backbone 的 `token_dim`。RankMixer–KuaiRand 的 target-attention 实际输入拼接了 **5 个 item 字段加 1 个 action 字段**，每个字段嵌入维数为 16，因此 `D=96`；其 `H=1`，辅助 `r=16`。不能遗漏 action 后误用 `D=80`，也不能误用 attention 输出维数 256。

| 设置 | 辅助分支活跃参数 | 模型 dense 活跃参数 | 状态 |
|---|---:|---:|---|
| B | 3,073 | 3,685,895 | 原始日志已核查 |
| C | 3,170 | 3,685,992 | 原始日志已核查 |
| D | 3,841 | 3,686,663 | 原始日志已核查 |
| E | 3,938 | 3,686,760 | 原始日志已核查 |
| L：token-local | 3,938 | 3,686,760 | 结构精确推导；CPU 插入点实例已检验；完整训练模型待预检 |
| F：input-independent | 3,082 | 3,685,904 | 结构精确推导；CPU 插入点实例已检验；完整训练模型待预检 |

B/C/D 关闭的张量仍可能储存在 checkpoint 中，但 `requires_grad=False` 且不进入前向。这些张量不是活跃容量。F 同理保留原 decay/phase 权重用于清晰继承初始化结构，但明确冻结，另引入每个 head 的 `r/2` 个常量相位增量。不得以 checkpoint 字节数或储存参数总数宣称容量匹配。

本次直接查询文件系统确认：

| 参照 | checkpoint | 字节数 |
|---|---|---:|
| B | `checkpoints/KuaiRand_Video_Action/SISA_components_RankMixer_KuaiRand_Video_Action_B_seed20262028_attempt1.model` | 2,067,198,921 |
| C | `checkpoints/KuaiRand_Video_Action/SISA_components_RankMixer_KuaiRand_Video_Action_C_seed20262028_attempt1.model` | 2,067,198,921 |
| D | `checkpoints/KuaiRand_Video_Action/SISA_components_RankMixer_KuaiRand_Video_Action_D_seed20262028_attempt1.model` | 2,067,198,921 |
| E，任务 548166_7 | `checkpoints/KuaiRand_Video_Action/SISA_single_seed_RankMixer_KuaiRand_Video_Action_sisa_seed20262028_ws2_bs16384_acc1_attempt1.model` | 2,067,202,270 |

注意，`rg --files checkpoints` 默认可能受 Git ignore 影响，不能把搜索不到 `.model` 当成权重已删除。本次保留状态由 `Path.glob` 与 `stat` 确认。

## 3. L：严格 token-local 且容量匹配的控制

令各 token 的局部辅助投影为 `b_t=W_b x_t`、`c_t=W_c x_t`，局部控制为

\[
g_t=-\operatorname{softplus}(W_a x_t+b_a),\qquad
\phi_t=W_\phi x_t.
\]

E 使用 `G_t=sum_{u<=t} g_u` 和 `Phi_t=sum_{u<=t} phi_u`。L 保留所有投影、参数初始化、head scale、head 数、辅助宽度、插入点与 Q/K/V 路径，用 `g_t, phi_t` 直接调制各 token，不进行 prefix scan。在不触发 exponent clipping 时，L 为

\[
\Delta^{L}_{ij}=\lambda_h\exp(g_i-g_j)
 c_i^\top R(\phi_j-\phi_i)b_j.
\]

因此在固定 `x_i,x_j` 时，其他 token 的表示如何变化，都不能改变这个辅助 pair score。这里的“局部”指 RISE 插入点接收到的表示，**不代表 backbone 表示本身没有历史信息**。对 RankMixer 此点接收到的是 item/action embeddings；若以后迁移到深层 contextual hidden，仍必须沿用这个定义。

数值细节必须显式处理：E 的 minimax centering 读整段 valid tokens。在无 clipping 时该公共 gauge 会从 query–key 乘积抵消；clipping 激活时则不一定抵消。L 将 gauge 固定为零，保留原 exponent 上下界 ±11，从而即使 clipping 激活，也不能通过 centering 引入非局部依赖。**E−L 的比较同时包含累积状态与相应稳定化方式的差异**；建议在验证集记录 clipping 命中率，若不能证明 clipping 足够稀少，则不得将全部差异写为纯粹的 prefix-scan 效果。

L 的 `g_i-g_j` 可能为正，因此不继承 E 在未裁剪 query reference 且 `j<=i` 时的非扩张保证。这个差异由移除累积而产生，应在方法定义中公开；不要把 local 公式也称为时间单调衰减。

可证伪假设 H-L：在同协议下，E 的 AUC 高于 L，且在具有更多有效历史的样本中优势更大。若 E≈L，结果说明在该场景下 token-local 的可学习变换已能解释收益，无法宣称累积是必要条件；若 L 优于 E，必须讨论累积可能引入过强或错误的跨 token 调制。任何一项结果都应保留，不以测试集结果决定是否展示。

## 4. F：输入无关但可学习的控制增量

F 保留原内容投影和原累积方式，改为每个 head 共享于所有 token 的

\[
g_t=-\operatorname{softplus}(a_h),\qquad
\phi_t=\omega_h\in\mathbb R^{r/2}.
\]

`a_h` 初值为原 decay bias −5，`omega_h` 初值为零，二者均由同一训练目标学习。有效位置计数仍受 mask 控制，因此控制状态依赖有效长度和位置，但不依赖该样本的内容表示。query reference 下、未触发 clipping 时，pair 调制退化为一种相对有效位置变换：

\[
\Delta^{F}_{ij}=\lambda_h\exp((n_i-n_j)g_h)
 c_i^\top R((n_j-n_i)\omega_h)b_j.
\]

这里 `n_t` 是包括当前位置的有效 token 计数；target 本身也按当前实现拼接在历史末尾。因此 F 不是固定无衰减、不旋转的弱退化 B，而是允许数据学习一个通用的衰减和旋转计划。它与外部 ALiBi 或 RoPE 算法并不相同，不应用这些算法名称替代 F。

可证伪假设 H-F：E 优于 F，支持样本相关的控制对当前比较有用。若 F≈E，则通用的位置变换已解释相当部分收益。**F 比 E 少 856 个活跃参数，且相位初始化不同，E−F 不能单独排除容量或优化差异。** L 对照帮助排除“只要增加同样多活跃参数就够了”的解释，但两组比较合起来仍不构成输入适应性效应的严格因果识别。F 初始化零相位是事前固定选择，不能依据 test AUC 改动初始化或增添候选后只报告赢家。

F 的“fixed”意为对输入固定，不是整个训练期间数值冻结。论文应写 `learned input-independent controls`，避免读者误以为衰减和相位未训练。

## 5. 最小任务矩阵和比较计划

| 任务 | 数据/模型 | 变更 | seed | full E 参照 | 执行状态 |
|---|---|---|---:|---|---|
| 0，L | RankMixer–KuaiRand | token-local 控制；所有原控制参数继续学习 | 20262028 | 548166_7 | 代码/配置/CPU 检验完成；本子任务未提交 |
| 1，F | RankMixer–KuaiRand | input-independent learned increments；保留累积 | 20262028 | 548166_7 | 代码/配置/CPU 检验完成；本子任务未提交 |

历史参照 E macro AUC 为 0.8615015；B 为 0.8543160；C 为 0.8617185；D 为 0.8614160。新增 L/F 未运行，表中不得预填推测性能。这不是一个已知 E 远胜其他模块的“挑好看的”场景：C 在既有单种子结果上略高于 E，本来就要求谨慎解释 AM、PA 的必要性。

确认性主要指标：标签等权 macro AUC。并列报告 6 个动作的 AUC、logloss，不能只报平均收益。预定两项对比 E−L、E−F；如果计算 p 值，对这两个主要比较控制多重检验。对历史长短的交互分析应以 effect difference 为统计对象，不能以“长历史显著、短历史不显著”代替组间差异检验。若使用用户级 bootstrap，按完整用户块重采样；预测样本必须依据完全一致的 row IDs 对齐。

保留 E/B/L/F 的同一批预测，可在主报告已有分组工具上分析 gain 与有效长度的关系。新训练任务有独立模型随机性；同一 seed 的用户级 bootstrap 仅反映给定训练模型下的测试抽样不确定性，不提供跨训练 seed 的重复证据。

## 6. 冻结协议、独立实现和启动命令

配置来源：`experiments/ablations/sisa_components_bcd/configs/base_configs.json` 与 `reference_params.json`。两份文件已逐字复制到本 study 的 `configs/rankmixer_controls_base_configs.json`、`configs/rankmixer_controls_reference_params.json`。每次运行先比对当前 YAML 全字段，再比对有效参数与原 E 日志全部字段；只允许 run ID 不同。控制语义作为 wrapper manifest 的独立字段保存，不误写进通用 `sisa_variant`。

冻结内容包括：2×H100 80GB、每卡 16,384、global batch 32,768、无梯度累积、1 epoch、history 上限100、base seed20262028、loader seed2027、aux-init seed20260822、embedding16、token256、BF16、现有 torch.compile、dense AdamW 1e−4、sparse Adagrad0.05、原 loader/monitor/coverage 规则。精确 GPU 型号并非论文接受比较的必要条件，但本最小研究复用现成 H100 协议减少额外异质性；若资源原因更换硬件，需另行形成透明的协议记录。

已新增且仅供本 study 使用：

| 文件 | 职责 |
|---|---|
| `scripts/controls.py` | L/F 的 `SISAScoreBias` 子类；`patched_build` 上下文只在当前进程更换 config factory |
| `scripts/run_rankmixer_controls.py` | 校验 frozen config；生成 source hashes/manifest；torchrun 两进程；在模型构建和 compile 前 patch，再 `runpy` 原 `run_expid.py` |
| `scripts/submit_rankmixer_controls.sbatch` | 仅数组 0=L、1=F，默认串行 `%1`；不修改/取消其他 agent 的作业 |
| `tests/test_controls.py` | 语义、参数、梯度、初始化、patch 恢复、adapter 集成与协议回归检验 |

以上相对路径均位于 `experiments/rise_sample_analysis/`。不修改 `unirank/`、`model_zoo/`、根目录 `run_expid.py`、通用 `scripts/`、`config/` 或 score_dim 实验。

可复现验证：

```bash
.venv/bin/python -m unittest discover -s experiments/rise_sample_analysis/tests -p test_controls.py -v
.venv/bin/python experiments/rise_sample_analysis/scripts/run_rankmixer_controls.py --control local --dry-run
.venv/bin/python experiments/rise_sample_analysis/scripts/run_rankmixer_controls.py --control fixed --dry-run
bash -n experiments/rise_sample_analysis/scripts/submit_rankmixer_controls.sbatch
```

供主 agent review 后使用的正式提交命令（本子任务未执行）：

```bash
sbatch experiments/rise_sample_analysis/scripts/submit_rankmixer_controls.sbatch
```

预计输出：`checkpoints/KuaiRand_Video_Action/RISE_controls_RankMixer_KuaiRand_Video_Action_{local|fixed}_seed20262028_attempt1.{log,model}`；manifest/outcome 写入 `artifacts/rise_rankmixer_controls/<run_id>/`。重试必须递增 `RISE_CONTROL_ATTEMPT`，不得覆盖已有结果。训练输出可能仍使用通用参数名 `sisa_enabled=True`，必须同时保留 manifest 的 `control` 字段和 code hashes，避免收集器将 L/F 错认成 full E。

加载 F 权重前必须再次构造 F 模块，因为含 `fixed_phase_increment`；加载 L 权重虽然 tensor keys 与 E 相同，但也必须构造 L，不能因 `load_state_dict` 成功就认为执行的是正确语义。不要在 compile 后替换子模块或在多个线程中动态切换 factory。

运行验收必须检查：两 rank 启动和步数推进；最后 `Test evaluation` 的 6 标签全指标；Slurm terminal state 与 return code；原始数据 coverage 一致；manifest 的控制类型和活跃参数数；checkpoint 保留；预测 row IDs 与 E/B 完全对齐。`dry-run` 或训练结束提示本身不足以证明性能结论成立。

## 7. 为什么本轮不直接在 HyFormer–MerRec 加一个“去 cumsum”

HyFormer 有三层 sequence self-attention 与三层 query-to-history attention，共六个插入点。前者使用 `sequence_end`，后者使用 `query`。把 `prefix_sum` 换成 identity 后仍沿用 `sequence_end`，query 控制继续读取其他 token，不能称为严格 local-only。此外，原实现用 `amin(cumulative_decay)` 找 endpoint 是因为累积 log-decay 单调；改成未累积的 log-decay 后 `amin` 不再等于最后有效位置，机械替换会额外改变参考选择规则。

未来若要在 HyFormer 定位此因素，需要显式区分 E、保留真实末端参考的 no-scan bridge、严格 query-reference local 三者，同时量化 clipping。当前两个 RankMixer 任务避免了这一额外比较，也节约资源。HyFormer 既有 B/C/D/E 可继续用于用户分组和冻结权重机制测量，其结论与这里的从头训练控制分开报告。

## 8. 公平 latency / memory benchmark 的实施规范

目标是填补论文附录 B.4 当前只给理论复杂度、缺实际开销的不足。基准应回答“当前仓库实现付出多少可观测成本”，不要自动承诺更快或部署收益。

### 8.1 两种测量层次

1. **插入点微基准**：至少 RankMixer 的 target-attention (`D=96,H=1,d_head=256,Nq=1,Nk=L`) 与 HyFormer 的 sequence-attention (`D=256,H=2,d_head=128,Nq=Nk=L`)。比较 native、B、E，L/F 可作为机制开销附加列。这两种形状区分线性长宽的打分与方阵打分；不能用只有一条 query 的低开销结果代表整个多层 self-attention backbone。
2. **完整模型稳态推理**：同一设备、同一固定样本批、同一精度分别测量 Native/B/E；优先使用已存在权重场景。Native 权重若已缺失，可在结构成本基准中使用固定随机初始化 native 模型，但必须标为 architecture-only timing，不能把这些输出用于 AUC。稀疏 embedding 查表与数据管道可能占主导，因此同时给出“GPU resident batch model latency”和另列的“含 CPU loader 的端到端吞吐”，二者不能混合。

训练效率另测同配置 `forward+backward`；含 optimizer 的整步开销应包含 dense/sparse 两种优化器和已初始化 optimizer state。GPU inference benchmark 用一张卡是测量设置，不是补造单卡正式实验结果。若要讨论双卡训练吞吐，须在正式 2 卡配置和同步范围下测量。

### 8.2 固定配置与计时

建议最小 shape 组：微基准 `B=256,L∈{20,100,200}`，完整模型 `B∈{256,2048},L=100`。`L=100` 为论文正式长度；其余仅作成本外推。若现有 model 固定 max_len，不应通过改变训练参数暗示重新训练过长序列。所有变体对同一 shape 成功后才进入该格比较，OOM 如实记录，不为某变体单独减 batch 后比较延迟。

同一 GPU、相同 torch/CUDA/driver、BF16/autocast、线程数、mask形状/valid比例、eval/train模式、TF32/deterministic设置。记录 GPU 型号、GPU UUID、节点、torch 版本、CUDA构建、driver、功耗/时钟、其他进程、输入dtype、输出shape与 finite检查。使用独占申请或确认设备未共享，不能与 score_dim agent 在同一张卡并发测量。

Eager 与 compiled 分开报；不把一个变体的 eager 与另一个的 compiled 比较。编译在计时前完成，并记录首次编译耗时，随后至少 30 次 warm-up；固定 shape 后每变体 10 个测量 block，每 block 100 次，轮换变体顺序。GPU resident 测量用 CUDA events，block边界同步；host end-to-end 另用 wall clock 并在边界同步。报告每个 block 平均值形成的 median 和 IQR，及 ratios；这些 block 不是 10 次独立训练，不能包装为跨 seed 置信区间。Batch latency p95 若要报告，应逐批计时，而非把block均值的p95标成请求tail latency。

不把文件读取、checkpoint 反序列化、编译、缓存建立混入 steady-state forward。若吞吐包含数据读取，则单独说明含什么；不能在 E 的同一热 cache 与 native 的冷 cache 下对比。

### 8.3 显存与 kernel 路径

各变体最好由独立子进程测量，避免 allocator cache 和 compiled graph 在变体之间残留。warm-up 后同步，记录静态 `memory_allocated`，reset peak，再执行固定数量的测量步；报告 total peak allocated、peak−baseline incremental allocated、peak reserved。仅 reset peak 不会清掉旧 graph 和 cache；不能把 reserved 误写成活跃张量内存。训练 memory 必须先初始化 optimizer state，再测 steady-state peak。

仓库 `ScaledDotProductAttention` 的 softmax 路径实际调用 `torch.nn.functional.scaled_dot_product_attention`。当 RISE 激活时，score bias 作为浮点 `attn_mask` 传入，native 可传布尔mask或无mask。必须记录 profiler 中实际 SDPA kernel 选择；一个 bias/mask 的加入可能改变 backend 选择，这就是当前实现成本的一部分。不要在没有 profiler 证据的情况下宣称所有变体都使用同一 FlashAttention kernel，也不要单独强制 native 的较慢 kernel 后称 RISE 开销很低。

论文所述 Q/K augmentation 是数学等价实现途径；当前已核查的这些 adapter 构造显式 score_bias，并不是所有位置都使用 augmented Q/K。benchmark 必须测当前实际实现。可以把融合版本另列为未来优化，但不能将未实现融合的理论开销替代实测。

### 8.4 报告格式与可写入论文的条件

| Model/site | shape B×Nq×Nk | precision/mode | Native ms | B ms | E ms | E/native | Native peak MiB | E peak MiB | backend | 完成状态 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| RankMixer target site | 待实测 | BF16/eager 或 compiled | — | — | — | — | — | — | 待 profiler | 本子任务仅设计 |
| HyFormer sequence site | 待实测 | BF16/eager 或 compiled | — | — | — | — | — | — | 待 profiler | 本子任务仅设计 |

输出 CSV/JSON 保存原始每 block时间、sample manifest、环境与代码哈希，不只保留截图。没有 GPU 实测之前，论文只能写理论成本与测量计划；本地登录节点没有 `nvidia-smi`，本次不提供伪造的 latency/memory 数值。

## 9. 当前实现检验记录

本次新增的 study-local `unittest` 验证以下必要语义：full factory patch 完整保留原 class/state/output/global RNG；L 的所有参数活跃且与 E 初始化逐张量相同；固定 pair 端点后修改其余输入/valid mask 不改变 L，Jacobian 在非端点严格为零；相同改变能影响 E；L 每个参数 tensor 都有有限非零梯度；F 的控制增量对输入值严格不变且活跃参数准确；F 的活跃参数获得梯度、冻结参数无梯度；padding/query/key slices 正确；factory在退出后恢复；拒绝 sequence_end 与已有 component selector 的组合；真实 RankMixer 插入点的 D=96/参数数量/原生权重初始化和输出形状正确；两个任务全部 frozen E 字段匹配。

这些是实现正确性的证据，不是 AUC、收敛性、GPU compile 或 DDP 吞吐证据。正式启动前仍应由主 agent 进行当前 workspace 的完整CPU回归及 Slurm 语法检查；本文件不对其他 agent 同时新增的代码作未经检查的通过声明。

本次最终执行结果：9 个 study-local CPU tests 全部通过，测试主体用时 0.257 秒；L/F 两个 dry-run 均成功；新增 Slurm 脚本 `bash -n`、两个 Python 实现 `py_compile` 均通过。两份冻结配置与原组件研究文件逐字节相同，其 SHA256 分别为 `24ce1a989ec820a90a14ea997c99929f90360d14aca07469d33d2800a5bef474`（base configs）和 `662177557fceb1a59bf5922ab013ab3fcf5f88ed57d7fd7836916ab7303d44da`（reference params）。

## 10. 新结果到达后的论文措辞规则

若 E>L 且 L 的容量完全匹配，可写“在此同协议单种子场景，跨 token 累积相较于等容量的局部调制提供了额外收益”，同时附 clipping 与历史软件环境限制。不得写“证明所有 backbone 必须累积”。若 F接近E，应明示输入无关的相对位置控制已经达到相近性能，不能隐去该对照维护主张。

效率结果无论优劣都要给绝对量和倍率。如果 GPU resident 开销明显高于参数比例所暗示的开销，讨论 prefix scan、浮点 attention bias、kernel 选择和 sequence layout；小参数量不等于低实际延迟。如果完整模型查表成本稀释了模块开销，应同时呈现微基准与模型基准，避免把 embedding 主导系统的百分比推广到所有部署场景。

本设计提供可被结果否定的主张，以及实现、协议、统计解释的边界；不会把增加对照数量本身当成学术贡献。
