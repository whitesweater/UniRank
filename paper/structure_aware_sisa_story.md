# 统一排序论文故事：A Unified Token Space Is Not a Unified Order

> 文档性质：论文叙事与研究设计备忘录，独立于正式论文正文。
>
> 当前状态：候选主线。已经由代码审计支持的问题事实，可以写成 motivation；涉及性能、因果机制和优越性的结论，必须经过下述判定性实验后才能写成论文 claim。
>
> 核心原则：完全删除 latent reasoning 叙事，把论文收敛到 unified ranking 中的结构感知状态化信息选择。

## 1. 两篇参考论文分别提供什么

### 1.1 SISA：机制来源，而不是我们的主问题

- 论文 PDF：<https://arxiv.org/pdf/2606.02332v2>
- HTML：https://arxiv.org/html/2606.02332v2

SISA 的核心问题是语言建模中 attention 与 SSM 的能力互补：attention 能进行全局内容检索，但标准 QK score 没有专门的 sequential-importance 通道；SSM 能累计状态，却不能像 attention 一样重新访问任意历史位置。SISA 将两者在 pre-softmax score 层融合：

$$
s_{ij}^{\mathrm{SISA}}
=
\frac{\mathbf q_i^\top\mathbf k_j}{\sqrt{d_h}}
+
\lambda_h\bar{\mathbf C}_i^\top\bar{\mathbf B}_j,
\qquad i\ge j.
$$

其状态通道通过输入相关的 decay 和 phase 构造：

$$
g_t=\sum_{u\le t}\log\alpha_u,
\qquad
\Phi_t=\sum_{u\le t}\theta_u.
$$

再将低维状态通道拼入 Q/K，以一次标准 SDPA 实现 score-level fusion。

因此，下列内容属于 SISA 已有贡献，不能作为本文的一级创新：

- compatibility 与 importance 的区分；
- SSM signal 进入 attention score；
- augmented Q/K 的单次 SDPA 实现；
- score-level fusion 相对于 block-level/head-level fusion 的定位。

本文可以使用 SISA 作为技术起点，但必须回答一个 SISA 在纯 causal language sequence 中没有面对的问题：**当输入不是一条纯序列，而是由有序行为、异构字段、目标和任务 token 共同组成时，累计状态究竟应沿什么结构传播？**

### 1.2 SpecFormer：问题发现和论文叙事范式

- 论文 PDF：<https://arxiv.org/pdf/2607.24025v2>
- HTML：https://arxiv.org/html/2607.24025v2

SpecFormer 最值得借鉴的不是具体频谱方法，而是它的叙事结构：

1. 先展示 Transformer 在推荐中的反常失败，而不是直接介绍模块；
2. 将失败定位为 recommendation-specific 的 embedding/attention collapse；
3. 给出“数据异质性与长尾分布 → embedding collapse → attention collapse → 深度退化”的机制链；
4. 方法中的每个组件都对应机制链中的一个环节；
5. 用 effective rank、层数扩展和线上结果同时验证性能与解释。

本文也应采用相同的证据顺序：

> 先证明 unified ranking 中存在可测量的结构性失败，再提出 SISA 的 ranking-compatible 改造；不能先有 SISA 模块，再为它寻找一个宽泛的 attention 缺陷。

## 2. 最终一级命题

### 2.1 最强的一句话

> **A unified token space is not a unified order.**

中文：

> **统一表示空间，不意味着所有 token 共享同一种顺序结构。**

辅助句：

> **Global interaction should not globalize temporal order.**

中文：

> **全局交互不应把时间顺序全局化。**

### 2.2 论文真正研究的问题

统一排序模型同时接收：

$$
\underbrace{\mathcal F=\{f_1,\ldots,f_m\}}_{\text{typed but non-sequential fields}},
\qquad
\underbrace{\mathcal S=(s_1,\ldots,s_L)}_{\text{chronological behaviors}},
\qquad
\underbrace{\mathcal V=\{v_1,\ldots,v_n\}}_{\text{target fields}}.
$$

统一架构通常将它们映射到共同 token 空间，并按照某个实现布局打包：

$$
\pi(\mathcal F,\mathcal S,\mathcal V)
=(z_1,\ldots,z_N).
$$

普通 attention 可以把这条轴主要看成 token 枚举；但是 cumulative state operator 会把它解释为有方向的路径：

$$
g_t^{\pi}=\sum_{u\le t}\log\alpha(z_u).
$$

于是，原本只是工程约定的 packing order 会进入模型语义。例如：

- device token 是否位于 age token 之后，开始影响累计 decay；
- target-category token 会继承此前 user/context token 的 phase；
- task token 会被当成行为路径的后续状态；
- 人为切分的 latent chunk 会被解释成连续状态转移。

我们将这种现象称为 **false chronology（伪时间关系）**；将 packing layout 对累计状态的非预期影响称为 **order leakage（顺序泄漏）**。

### 2.3 为什么这个问题足够“痛”

这个问题不只是“attention 还能不能再提高一点”，而是同时影响：

1. **语义正确性**：状态传播不再对应真实行为时间；
2. **模型稳健性**：结果可能依赖字段声明、tokenizer 切块和 adapter packing；
3. **机制解释**：性能增益无法被可靠归因于 cumulative behavioral importance；
4. **跨架构迁移**：同一个 SISA 模块在不同模型中扫描的对象不同，所谓“统一插件”实际实现了不同归纳偏置；
5. **科学可证伪性**：如果不控制 packing、scan scope 和参数容量，主结果只能证明“增加一个低维 score bias 有时有效”。

最具审稿人冲击力的问题是：

> Why should the state transition from device type to target category depend on which field is serialized first?

## 3. 当前实现提供的事实基础

### 3.1 公共 SISA 模块的行为

当前公共实现接收 `hidden_states[B, L, D]`，生成 `log_alpha[B, H, L]`，然后沿完整 `L` 轴累计 decay 和 phase：

$$
\mathbf g=\operatorname{PrefixSum}_{L}(\log\boldsymbol\alpha),
\qquad
\boldsymbol\Phi=\operatorname{PrefixSum}_{L}(\boldsymbol\theta).
$$

实现没有 token type、segment id 或 boundary reset。`valid_mask` 只会将无效位置的贡献置零，不会建立新的累计段。`query_slice` 和 `key_slice` 在完整累计之后才应用，因此不能隔离不同 token group。

代码位置：

- `unirank/pytorch/layers/attentions/sisa.py:233-280`
- `unirank/pytorch/layers/attentions/sisa.py:318-339`

### 3.2 各模型的实际 scan scope

| 模型 | 当前累计范围 | 结构判断 |
|---|---|---|
| RankMixer | `[history, target]` | 不含 user/context，但 target 仍进入同一累计轴 |
| OneTrans | `[history, NS(user/context/target latent tokens)]` | 明确跨越有序与非序列结构 |
| HiFormer | target attention 扫 `[history,target]`；统一层再扫 feature/task tokens | feature、summary、task token 被放入单段累计 |
| Zenith | target attention 扫 `[history,target]`；统一层扫 ID/context/attribute chunks | 非序列与 summary chunks 被时间化 |
| HyFormer | sequence encoder 只扫 behavior；query decoder 扫 `[behavior, global tokens]` | 部分正确，部分跨结构 |
| UltraHSTU | `[user aggregate, history, candidate]` | user 和 candidate 与行为共用状态轴 |
| UniMixer | 所有输入 flatten 后切成 latent chunks，再扫全部 chunks | scan 甚至不对应原始行为时间轴 |

因此，当前代码可以支持如下 motivation：

> Existing direct adaptations perform cumulative scanning over adapter-specific packed axes rather than an explicitly defined behavioral structure.

但当前代码不能支持如下方法 claim：

> The state scan is restricted to ordered behaviors.

这必须在实现修改和实验完成后才能写入正式论文。

### 3.3 reference 也不是统一的

当前代码不是统一采用 sequence-end reference：

- OneTrans、RankMixer、UltraHSTU 的相应站点使用 query reference；
- HiFormer 和 Zenith 同时包含 query-reference target attention 与 sequence-end unified attention；
- HyFormer 的 sequence encoder 使用 sequence-end，而 query decoder 使用 query reference；
- UniMixer 使用 sequence-end。

所以正式论文不能把 sequence-end 写成当前方法的统一设计。新的结构感知方法应先定义 query type，再明确每一类 query 的 reference。

## 4. 推荐的方法：Structure-Aware State-Informed Attention

工作名称可暂定为：

- **Structure-Aware SISA（SA-SISA）**；
- **Typed State-Informed Attention（TSIA）**；
- **Order-Respecting State-Informed Attention**。

在方法和实验稳定前，不急于锁定缩写。

### 4.1 原则一：状态只由真实行为链产生

先显式抽取按时间排序的行为 token：

$$
\mathbf H_{\mathcal S}
=(\mathbf h_1^{\mathcal S},\ldots,\mathbf h_L^{\mathcal S}).
$$

仅在该轴上计算：

$$
g_t^{\mathcal S}
=\sum_{u\le t}\log\alpha_u,
\qquad
\Phi_t^{\mathcal S}
=\sum_{u\le t}\theta_u.
$$

user/context/target/task tokens 不贡献 decay 或 phase。实现上应当 gather behavior tokens、完成 scan，再将状态通道 scatter 到对应 attention positions；不能仅依靠在 packed axis 上把非行为位置置零，因为外部 query 的 reference 仍可能随其 packing 位置变化。

### 4.2 原则二：所有 token 保留 compatibility interaction

普通 QK score 不需要限制在行为边：

$$
s_{ij}^{\mathrm{compat}}
=\frac{\mathbf q_i^\top\mathbf k_j}{\sqrt{d_h}}.
$$

这保留 unified ranking 的核心价值：field、target、task 和 behavior token 仍然能够全局交互。

### 4.3 原则三：state channel 只调制行为证据

推荐的统一形式是：

$$
s_{ij}
=
\frac{\mathbf q_i^\top\mathbf k_j}{\sqrt{d_h}}
+
\mathbf 1[j\in\mathcal S]\,
\lambda_h
\left(\bar{\mathbf c}_i^{\,r(i)}\right)^\top
\bar{\mathbf b}_j
+M_{ij}.
$$

其中，indicator `j in S` 表示 cumulative importance 是历史行为证据的属性。对于非行为 key，state term 为零，仍由 QK compatibility 决定其权重。

该定义实现了清晰分工：

- **QK channel**：谁和谁在内容或特征上匹配；
- **state channel**：某条历史行为证据经过后续行为后保留了多少重要性；
- **hard mask**：哪些交互边被允许；
- **token type/reference policy**：不同 query 如何读取行为状态。

### 4.4 原则四：reference 由 query 类型决定

建议先采用最容易解释的 typed-reference policy：

1. behavior query 使用当前位置状态 `r(i)=i`；
2. target、field、global 或 task query 使用行为序列末端状态 `r(i)=L`；
3. 无行为序列时使用零状态或显式 no-history state；
4. target-token reference 作为独立消融，不预设一定优于 sequence-end。

这样，外部 query 对历史的读取不会因它在 packed tensor 中位于行为前还是行为后而改变。

### 4.5 原则五：保持原 backbone 与计算路径

方法仍应满足：

- 不替换原始 Q/K/V；
- 不改变 residual、FFN、tower 和 loss；
- 不重新打开 hard-masked edge；
- 尽量继续使用 augmented Q/K 和标准 SDPA；
- 明确报告额外参数，并设置 parameter-matched control；
- 对 FlexAttention 等站点保留其原始 sparse mask。

## 5. 各 backbone 的适配策略

### 5.1 RankMixer

- 只在已有 target-to-history attention 中加入状态通道；
- scan 输入仅为 history；
- target query 使用 sequence-end reference；
- 后续纯 token mixer 不加入 SISA。

这是最干净的 behavior-only 基线。

### 5.2 HiFormer 与 Zenith

- target-attention pooling：使用 behavior-only scan；
- 后续 field/task self-attention：保留普通 QK，不对纯字段 token 做累计；
- 可以设置“unified all-token SISA”作为错误结构或宽松结构对照，而不是默认方法。

### 5.3 HyFormer

- sequence self-attention：按 behavior query reference 计算；
- global-query-to-sequence cross-attention：history keys 使用累计状态，global queries 使用 sequence-end reference；
- NS/query-boosting token 不进入 scan。

它最适合展示“行为状态与全局 query 在 score 层融合”。

### 5.4 OneTrans

- sequence 与 NS token 仍在原 attention 中全局交互；
- cumulative state 从 sequence stream 单独计算；
- 只为 sequence keys 提供 state channels；
- NS query 使用 sequence-end reference，NS key 不使用 cumulative state。

### 5.5 UltraHSTU

- user aggregate token 和 candidate token 不进入行为 scan；
- history 形成独立状态轴；
- candidate query 使用 terminal behavior state；
- sparse causal/local mask 保持不变。

### 5.6 UniMixer

UniMixer 在 tokenizer 前已经 flatten user/context/history/target，随后按连续维度切成 latent chunks。此时原始行为边界可能不可恢复。

因此只有两种诚实选择：

1. 修改 tokenizer，使行为 token/group identity 在 mixing 前保留；
2. 将 UniMixer 作为当前方法的适用边界，不宣称 behavior-only state operator 可以无改造插入所有 unified backbones。

不应把任意 latent chunk order 解释为行为时间。

## 6. Introduction 的六段故事结构

### 第一段：统一排序的趋势与价值

现代 ranking 同时依赖用户/上下文字段、目标物品和行为历史。token-based unified architectures 将这些来源映射到共同表示空间，使 sequence modeling 与 feature interaction 能够共享高吞吐算子并进行早期交互。

落点：

> Unified tokenization has become a powerful interface for jointly modeling heterogeneous features and behavioral histories.

### 第二段：共同空间掩盖了不同结构

这些输入虽然具有共同维度，却不具有共同拓扑：行为是有时间方向的链，字段和目标属性是有类型但非时序的集合。统一表示解决了维度兼容，却没有自动解决结构兼容。

落点：

> Representation unification does not imply order unification.

### 第三段：为什么标准 attention 与直接状态化都不完整

标准 QK attention 可以进行全局 feature compatibility 建模，但没有专门的累计行为状态通道。SISA 表明 cumulative state 可以直接进入 attention score；然而它原本建立在纯 causal token sequence 上。若直接把该 scan 应用于 unified ranking 的 packed axis，就会把字段、目标、任务和 latent chunks 纳入同一状态路径。

落点：

> A state mechanism can recover sequential importance while simultaneously introducing false chronology when the scanned axis is only a serialization convention.

### 第四段：核心研究问题

统一排序真正需要的是：

> How can a ranking model use cumulative behavioral dynamics to route information without imposing temporal transitions on non-sequential tokens?

这将问题从“加一个 attention bias”提升为“如何在一个统一算子内同时尊重 set-like 与 sequence-like 结构”。

### 第五段：本文方法

提出 structure-aware state-informed attention：普通 QK score 保持全 token compatibility；decay/phase 只沿 behavior sequence 累计；state term 只作用于行为证据边；不同类型 query 使用显式 reference policy。该设计在 score 层融合全局检索与行为状态，同时保持原 backbone 和 hard mask。

### 第六段：证据闭环

实验不只比较最终 AUC，而是回答：

- false chronology 是否可测量；
- behavior-only 是否优于 all-token scan；
- 收益是否来自累计动态而非额外参数；
- reference 与 edge scope 分别起什么作用；
- 哪些架构保留了足够清晰的行为结构；
- 方法是否在更长历史上更有优势且成本可接受。

## 7. 三项贡献的推荐写法

### Contribution 1：问题与诊断

> We identify false chronology in unified ranking: applying cumulative sequence dynamics to a packed mixture of ordered behaviors and non-sequential tokens introduces structurally unjustified dependencies on token layout.

在没有完成 packing intervention 和诊断实验前，使用 “identify and study” 而不是 “prove a universal failure”。

### Contribution 2：方法

> We propose a structure-aware state-informed attention operator that preserves all-token compatibility while restricting cumulative dynamics to ordered behavioral evidence and explicitly controlling which attention edges receive state modulation.

### Contribution 3：证据

> We conduct controlled cross-architecture experiments that isolate scan scope, state-affected edges, reference selection, packing layout, and parameter capacity under a unified ranking protocol.

## 8. 判定性实验设计

### RQ1：false chronology 是否真实存在

#### Scan-only packing intervention

保持 backbone 的 token identity、普通 QK score 和输出位置不变，仅改变非序列 token 进入 cumulative operator 的枚举次序，再将结果映射回原位置。

报告：

- prediction difference；
- state-bias matrix difference；
- AUC/logloss 波动；
- 不同层的 order sensitivity。

这个实验最直接隔离 cumulative operator 的顺序依赖。

#### Semantic-preserving layout intervention

训练或评估多个语义等价 packing layout，同时同步重排：

- field identity/metadata；
- field-specific tokenizer 或投影参数；
- token mask 和输出映射。

不能直接交换 gender 与 device 的取值，也不能只 shuffle field-specific token 而不重排其专属参数；那会制造 OOD 输入，而不是检验 serialization invariance。

建议定义：

$$
\mathrm{OSS}
=
\mathbb E_{x,\pi,\pi'}
\left|
\hat y_{\pi}(x)-\hat y_{\pi'}(x)
\right|,
$$

作为 Order Sensitivity Score。

### RQ2：状态应该在哪里扫描、作用到哪里

主消融：

1. Base；
2. All-token SISA（当前直接适配）；
3. Behavior-only scan；
4. Behavior-only scan + state on all keys；
5. Behavior-only scan + state on behavior keys only；
6. Non-sequential-only pseudo-scan。

其中第 6 项是重要的反事实：若非序列 pseudo-scan 同样有效，“行为累计语义”解释就不充分。

### RQ3：是否只是额外容量或低秩 bias

比较：

$$
\begin{aligned}
&\text{Base},\\
&\text{Base}+\lambda c_i^\top b_j
&&\text{no cumulative decay/phase},\\
&\text{Base}+\text{fixed positional decay},\\
&\text{Base}+\text{data-dependent decay},\\
&\text{Base}+\text{phase only},\\
&\text{Full structure-aware state channel}.
\end{aligned}
$$

必须同时提供 parameter-matched control。原始 SISA 通过缩减 FFN 保持参数预算相近；当前 UniRank 适配增加了参数而没有同等缩减，因此仅比较 Base 与 Full 不足以排除容量解释。

### RQ4：reference policy

比较：

- query reference；
- sequence-end reference；
- explicit target reference；
- no-reference/normalized control；
- typed reference（behavior=query，external query=sequence end）。

目标不是证明 sequence-end 永远最好，而是解释不同 query 类型为什么需要不同 reference。

### RQ5：跨架构边界

至少覆盖：

- target-attention pooling：RankMixer；
- stacked feature interaction：HiFormer 或 Zenith；
- layer-wise sequence/feature interaction：HyFormer 或 OneTrans；
- sparse sequential transduction：UltraHSTU。

UniMixer 只有在保留 behavior group identity 后才适合进入“同一方法”的主表，否则应作为边界分析。

### RQ6：长度与效率

主 benchmark 当前 `max_len=100`，建议分桶：

$$
[1,10],\quad[11,30],\quad[31,60],\quad[61,100].
$$

更长历史必须使用独立的 `max_len=128/256/512` 扩展协议，不能在长度上限为 100 的主实验中报告 `[200,+\infty)`。

同时报告：

- 参数量与 parameter-matched 参数量；
- FLOPs；
- samples/s；
- peak memory；
- end-to-end latency；
- state scan 和 augmented Q/K 的独立开销。

### RQ7：正向和负向顺序对照

- 打乱或反转 behavior order：性能和 state bias 应明显改变；
- 改变 non-sequential serialization：结构感知版本应基本稳定；
- 同时报告二者，可以证明模型不是简单地对所有顺序都不敏感，而是对正确的顺序敏感。

## 9. 最小验证路线与停止条件

在全面训练前，先选择 RankMixer 与 HyFormer 进行小矩阵验证：

$$
\text{Base}
\rightarrow
\text{All-token/current SISA}
\rightarrow
\text{Behavior-only}
\rightarrow
\text{Behavior-key-only}.
$$

每个设置至少加入：

- 两到三种非序列 packing intervention；
- 原始与 shuffled behavior；
- parameter-matched bias control；
- query 与 sequence-end reference。

支持该故事的最低证据应同时满足：

1. current all-token variant 存在显著 packing sensitivity；
2. behavior-only variant 显著降低该敏感性；
3. behavior-only 至少保持或改善主要效果；
4. shuffled behavior 会削弱收益，证明模型确实使用了 chronology；
5. matched low-rank bias 不能解释全部增益。

如果 all-token 始终更好、packing intervention 影响很小，或者 non-sequential pseudo-scan 同样有效，就不应强行采用 false-chronology 主线。此时应退回更保守的“ranking adaptation of score-level importance”定位。

## 10. 现有实验能够说明什么

已有三 seed 汇总覆盖 HiFormer、HyFormer、RankMixer 和 Zenith 的 16 个模型—数据集单元。当前 SISA 相对本地 baseline 的 cell-macro `Delta AUC` 为 `+0.006415`，其中 15/16 单元为正；详细记录见：

- `experiments/sisa_three_seed_unified/results/summary.md`
- `experiments/sisa_three_seed_unified/results/model_summary.csv`

这批结果可以作为 feasibility evidence：score-level state channel 在 unified ranking 中值得继续研究。

但它不能证明：

- 增益来自 behavior-only cumulative dynamics；
- all-token scan 的字段顺序是合理的；
- sequence-end reference 优于 query reference；
- 增益不是额外参数或一般 low-rank bias；
- 七个 adapter 实现的是同一种机制。

新论文应把现有结果作为 preliminary/full-plugin baseline，再用结构消融建立因果解释。

## 11. 明确禁止的叙事和表述

正式论文不应出现：

- latent reasoning / reasoning trajectory / reasoning frontier；
- test-time reasoning depth；
- “我们提出了 score-level fusion”；
- “Transformer 不能建模动态兴趣”；
- “所有字段本质可互换”；
- “当前实现统一使用 sequence-end reference”；
- “当前 state scan 已限制在 behavior sequence”；
- “无需适配即可插入任意 unified backbone”；
- “性能提升证明 cumulative state 具有推荐语义”。

更稳妥的替代表述：

- Transformers may learn sequential dynamics implicitly, but standard QK attention does not expose a dedicated cumulative-state channel.
- Recommendation fields are semantically typed, but their serialization order is not a temporal relation.
- We preserve field identity while removing cumulative dependence on field enumeration.
- We adapt, rather than originate, SISA-style score-level fusion.
- Our method applies to backbones that retain or can expose behavioral token identity.

## 12. 推荐标题

首选：

> **A Unified Token Space Is Not a Unified Order: Structure-Aware State-Informed Attention for Ranking**

备选：

> **Avoiding False Chronology in Unified Ranking Models**

> **Global Interaction without Global Temporalization: State-Informed Attention for Unified Ranking**

> **Respecting Fields and Behaviors in State-Informed Unified Ranking**

## 13. 审稿人最终应如何概括本文

英文：

> This paper identifies a structural mismatch in unified ranking: cumulative sequence operators applied to packed heterogeneous tokens can mistake serialization order for behavioral chronology. It introduces a structure-aware score operator that retains global QK compatibility while deriving state modulation only from ordered behavioral evidence, and validates the mechanism through layout, scope, reference, capacity, and cross-architecture controls.

中文：

> 本文发现，统一排序把字段、目标和行为放入共同 token 空间后，累计序列算子可能将人为 serialization 误当作行为 chronology。本文在保留全局 QK 交互的同时，只从真实行为链构建状态化 importance，并通过 packing、scan scope、edge scope、reference、参数容量和跨架构实验验证这一机制。

## 14. 最终叙事层级

全文应保持以下层级，不能倒置：

1. **一级问题：统一表示造成结构混同和 false chronology；**
2. **一级方法：将 ordered state propagation 与 heterogeneous global interaction 分离；**
3. **融合位置：在 attention score 中重新统一二者；**
4. **技术来源：采用并改造 SISA 的低维累计状态通道；**
5. **实验目标：证明结构约束、累计动态和 score-level integration 各自的作用。**

其中，“Compatibility is not importance”仍然可以作为解释 SISA score 的一句话，但不再承担整篇论文的主问题。本文真正有辨识度的命题是：

> **Unified tokens may interact globally, but only genuine behavioral evidence should evolve chronologically.**
