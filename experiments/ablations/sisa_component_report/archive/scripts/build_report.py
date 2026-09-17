"""Render this reviewed report snapshot from the strict component evidence.

Narrative conclusions are scoped to the three reviewed, complete groups.
Changed evidence requires a fresh interpretation review before publication.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


REPORT_DIR = Path(__file__).resolve().parents[2]
NAMES = {
    "sisa_components_bcd": "RankMixer–KuaiRand",
    "sisa_components_hyformer_merrec_bcd": "HyFormer–MerRec",
    "sisa_components_hiformer_taac_bcd": "HiFormer–TAAC-25",
}
VARIANTS = {
    "B": "仅内容打分",
    "C": "内容打分＋衰减",
    "D": "内容打分＋旋转",
    "E": "完整 SISA",
}


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def mean_metric(run, metric):
    assert run["eligible"] and not run["issues"], run["run_id"]
    values = [m[metric] for m in run["metrics"].values()]
    value = sum(values) / len(values)
    key = "macro_auc" if metric == "AUC" else "macro_logloss"
    assert math.isclose(value, run[key], rel_tol=0, abs_tol=1e-12)
    return value


def main():
    evidence = json.loads((REPORT_DIR / "archive/results/evidence.json").read_text())
    groups = {
        study: {r["setting"]: r for r in evidence["runs"] if r["study"] == study}
        for study in NAMES
    }
    complete = [s for s, g in groups.items() if all(g[k]["eligible"] for k in "BCDE")]
    assert complete == list(NAMES), "All three reviewed groups must be complete"
    reviewed = [groups[s] for s in complete]
    rank, hy, hi = reviewed
    assert mean_metric(rank["C"], "AUC") > mean_metric(rank["E"], "AUC")
    assert mean_metric(hy["E"], "AUC") > max(mean_metric(hy[k], "AUC") for k in "BCD")
    assert mean_metric(hi["E"], "AUC") > max(mean_metric(hi[k], "AUC") for k in "BCD")
    assert mean_metric(hi["D"], "logloss") < min(mean_metric(hi[k], "logloss") for k in "BCE")
    for g in reviewed:
        for label in g["B"]["metrics"]:
            assert all(g[k]["metrics"][label]["AUC"] > g["B"]["metrics"][label]["AUC"] for k in "CDE")

    macro_rows = []
    for study in complete:
        g = groups[study]
        best_auc = max(mean_metric(g[k], "AUC") for k in "BCDE")
        best_loss = min(mean_metric(g[k], "logloss") for k in "BCDE")
        for k in "BCDE":
            a, l = mean_metric(g[k], "AUC"), mean_metric(g[k], "logloss")
            macro_rows.append([
                NAMES[study], k + "：" + VARIANTS[k],
                f"**{a:.7f}**" if a == best_auc else f"{a:.7f}",
                f"**{l:.7f}**" if l == best_loss else f"{l:.7f}",
            ])
    macro_table = table(["模型—数据集", "变体", "宏平均 AUC ↑", "宏平均 LogLoss ↓"], macro_rows)

    contrast_rows = []
    for study in complete:
        g = groups[study]
        for lhs, rhs in [("C", "B"), ("D", "B"), ("E", "B"), ("E", "C"), ("E", "D")]:
            contrast_rows.append([
                NAMES[study], f"{lhs}−{rhs}",
                f"{mean_metric(g[lhs], 'AUC') - mean_metric(g[rhs], 'AUC'):+.7f}",
                f"{mean_metric(g[lhs], 'logloss') - mean_metric(g[rhs], 'logloss'):+.7f}",
            ])
    contrasts = table(["模型—数据集", "对比", "ΔAUC ↑", "ΔLogLoss ↓"], contrast_rows)
    parameter_rows = [[NAMES[s], *[f"{groups[s][k]['dense_parameters']:,}" for k in "BCDE"]] for s in complete]
    parameters = table(["组合", "B", "C", "D", "E"], parameter_rows)
    status_rows = [[NAMES[s], "3/3 已完成并通过核验", "已核验", len(groups[s]["E"]["metrics"])]
                   for s in complete]
    status_table = table(["组合", "新增 B/C/D", "历史 E", "每配置标签数"], status_rows)

    label_sections = []
    for study in complete:
        g = groups[study]
        label_sections.append("### " + NAMES[study])
        for metric, title in [("AUC", "AUC，越大越好"), ("logloss", "LogLoss，越小越好")]:
            rows = [[label, *[f"{g[k]['metrics'][label][metric]:.6f}" for k in "BCDE"]]
                    for label in g["B"]["metrics"]]
            rows.append(["任务等权宏平均", *[f"{mean_metric(g[k], metric):.7f}" for k in "BCDE"]])
            label_sections.extend([title + "。", table(["任务", "B", "C", "D", "E"], rows)])
    labels = "\n\n".join(label_sections)

    content = f"""# SISA 组件消融实验方案与结果报告

结果快照：{evidence['generated_at']}（北京时间）。

本报告覆盖三组已完成的组件消融：RankMixer–KuaiRand、HyFormer–MerRec、HiFormer–TAAC-25。每组新增 B、C、D 三次训练，完整 SISA 的 E 组复用预先指定的 H100 历史轮次。九个新增任务全部成功完成并通过核验，连同三个历史 E，共形成十二条正式运行记录、三组完整比较。正式结果均来自最终测试集，未使用训练或验证中间指标。

**主要发现：单独引入衰减或旋转，在三个组合的全部 13 个任务上均提高了内容打分对照的 AUC。完整 SISA 在 HyFormer–MerRec 和 HiFormer–TAAC-25 上取得最高宏平均 AUC；RankMixer–KuaiRand 的仅衰减变体取得最高 AUC。LogLoss 的最优配置分别为 RankMixer 的仅衰减、HyFormer 的完整 SISA、HiFormer 的仅旋转。组件组合收益具有场景及指标依赖性。**

## 1. 研究动机与消融设计

SISA 在骨干已有注意力的 softmax 之前增加一个分数偏置。原始 Q/K/V、已有掩码处理、残差和前馈结构保持不变。该偏置由内容投影、输入相关衰减、输入相关旋转以及每头可学习缩放构成。内容投影提供额外的匹配分数；衰减改变累积位置关系对应的权重；旋转改变投影通道之间的相对方向。实验考察这些结构化变换是否比单独增加内容打分更有效，以及两个变换联合使用能否带来额外收益。

| 组别 | 名称 | 内容打分 | 动态衰减 | 动态旋转 | 对应问题 |
| --- | --- | --- | --- | --- | --- |
| B | content_only | 开启 | 关闭 | 关闭 | 仅增加内容打分分支能达到什么效果？ |
| C | decay_only | 开启 | 开启 | 关闭 | 在内容打分基础上加入衰减是否有效？ |
| D | rotation_only | 开启 | 关闭 | 开启 | 在内容打分基础上加入旋转是否有效？ |
| E | full SISA | 开启 | 开启 | 开启 | 同时使用两个组件能否进一步改善？ |

内容打分是衰减与旋转共同作用的基础，故 B/C/D/E 均保留它。B 是这条额外分支的对照，并不等同于原始骨干 baseline。以衰减、旋转两个开关来看，这是一个 2×2 组件设计；每组只新增三个训练任务，是因为 E 已有匹配结果。

关闭衰减时令 log(alpha)=0，即 alpha=1；关闭旋转时令相位为零。所有分支保持原有初始化顺序，未启用的分支被冻结并跳过前向计算。共同参数使用相同初始化规则；每头可学习缩放始终保留。各变体从头训练，不从 E 的 checkpoint 微调。

预先规定 C−B 和 D−B 分别衡量单独加入衰减、旋转的收益；E−C 衡量已有衰减时再加入旋转的收益；E−D 衡量已有旋转时再加入衰减的收益。E−B 衡量完整结构相对内容打分对照的总差异。

## 2. 实验范围、数据与接入位置

本轮从已接受的四模型、四数据集主实验范围中选择以下三组，根据已知主实验收益进行机制分析。它们不是从全部 16 个组合中随机抽取的代表样本，不据此推断所有模型—数据集组合。

| 组合 | 标签 | SISA 接入位置 |
| --- | --- | --- |
| RankMixer–KuaiRand | is_click、is_follow、is_like、is_comment、is_forward、long_view | 原有 target-attention pooling；不替换 token mixer |
| HyFormer–MerRec | Like、Cart、Offer、Checkout、Purchase | 三层序列自注意力及三层 query-to-sequence attention |
| HiFormer–TAAC-25 | is_click、is_conversion | 目标注意力及三层融合 token 注意力 |

HiFormer 的融合 token 不等同于时间事件，因此不能将所有接入位置的作用统一解释为“遗忘较早的用户行为”。本轮没有单独改变接入位置，因而不区分不同位置的独立贡献。

所有组沿用已有 Parquet 数据、用户/物品侧表、标签构造和时间划分，不重做预处理。KuaiRand 训练区间为 2022-04-07 至 05-02，验证为 05-03 至 05-05，测试为 05-06 至 05-08。MerRec 训练为 2023-10-01 至 10-25，验证为 10-26 至 10-28，测试为 10-29 至 10-31；以每个 user–item 的首次 item_view 为样本锚点，将其后五类行为按 OR 合成为标签。TAAC-25 训练为 2024-10-15 至 2025-05-28，验证为 2025-05-29，测试为 2025-05-30 至 08-05；沿用按正样本数约束尾部划分、过滤无 click/conversion 用户及最少十次交互的规则。

下表为两卡协议中实际保留的交互样本数，由各 rank 的完整 batch 数乘以每卡 batch 后相加得到。train/valid/test 均沿用按数据块丢弃尾部不足一个 batch 的规则。它们不是独立用户数，也不同于预处理后全部数据行数。

| 数据集 | 训练保留样本 | 验证保留样本 | 测试保留样本 |
| --- | --- | --- | --- |
| KuaiRand | 255,836,160 | 34,258,944 | 32,964,608 |
| MerRec | 144,048,128 | 16,449,536 | 11,599,872 |
| TAAC-25 | 585,957,376 | 17,793,024 | 153,157,632 |

三组 B/C/D 的实际逐 rank 块数、行数和 batch 数均与各自 E 匹配，表中包含已完成 HiFormer–TAAC-25 的最终运行覆盖。历史数据未保存全量内容散列，覆盖一致不能独立证明样本值逐字节一致。本轮也未重新审计事件级标签归因；MerRec 标签归因窗口及是否跨划分边界仍应在论文数据说明中核实。

## 3. 固定训练与评估设置

每组 B/C/D 均匹配对应模型—数据集在 seed 20262028 的已有完整 SISA H100 轮次。对照在启动前固定，未根据消融结果挑选 E 的种子。原先四卡主实验与本轮存在 DDP 布局、随机流和尾批覆盖差异，因此其 baseline 不混入本轮组件差值。本次新增任务与所复用 E 均采用两张 H100。

| 设置项 | 固定值 |
| --- | --- |
| 硬件与并行 | 2×NVIDIA H100 80GB HBM3，DDP |
| 每卡 / 全局 batch | 16,384 / 32,768 |
| 梯度累积 / 训练预算 | 1 / 1 epoch |
| 最大历史长度 | 100 |
| 实验随机种子 | 20262028；各 rank 使用 seed＋rank |
| 数据加载随机种子 | 2027；epoch/rank 派生规则保持不变 |
| SISA 参数随机种子 | 20260822；接入位置偏移保持不变，隔离随机数生成器 |
| Embedding / token 维度 | 16 / 256 |
| 骨干层数 / 任务塔 | 3 / [128, 64]，ReLU |
| 稠密优化器 | AdamW，学习率 0.0001，weight decay=0 |
| 稀疏优化器 | Adagrad，学习率 0.05 |
| 损失 | 各任务二元交叉熵先按 batch 求均值，再等权相加 |
| 精度 / 编译 | BF16 / 启用 torch.compile |
| 梯度检查点 / net_dropout | 关闭 / 0 |
| 数据加载 worker | 每 rank 4 个 |
| SISA 分数通道维度 | 16 |
| SISA 缩放初值 / 固定 score_scale | 0.1 / 1.0 |
| SISA 数值默认值 | decay_bias=−5，exp_clamp=11 |
| 验证与模型选择 | epoch 末同步验证，按验证集任务宏平均 AUC 选择最佳 checkpoint |
| 最终测试 | 加载最佳 checkpoint，汇集各 rank 预测后逐标签计算指标 |

HiFormer 的融合 token 注意力为 2 heads，目标注意力汇聚为 1 head，qkv_rank=256；HyFormer 使用两个 heads、四个非序列 token 和两个 global token；RankMixer 使用 Chunk tokenizer 和四个非序列 token。不能把某个接入位置的 head 参数当作全部骨干统一配置。MerRec 的稀疏优化器额外沿用 foreach=False、chunk_size=16,777,216；其他组沿用各自 E 的默认实现。软件依赖由项目锁文件管理，Python 3.12、PyTorch 2.8.0，采用 CUDA 12.6 对应构建；历史 E 未保留本轮同格式的软件与全源码 manifest，不声称所有运行环境逐字节相同。

实际日志中的有效配置已逐项与历史 E 对比，允许变化仅为 sisa_variant 和区分输出的 model_id。score_dim、lambda_init、score_scale 由日志核对；decay_bias、exp_clamp 来自实现默认值，不伪称为旧日志显式记录字段。各新增任务保存了源码散列与 Git 版本，历史 E 的源码与软件环境记录边界如上所述。

每个任务报告 ROC-AUC（越高越好）和二元 LogLoss（越低越好）。组合内宏平均定义为 K 个标签指标的算术平均：Macro-AUC = (1/K)Σ AUC_k，Macro-LogLoss = (1/K)Σ LogLoss_k。先汇集样本计算各标签指标，再对标签等权平均；不把多任务标签混成一个二分类指标。所有差值均为绝对差，不是相对百分比。仅一个实验种子，不报告标准差、置信区间或显著性星号。

## 4. 三组正式测试结果

表中加粗为同一组合内最好的数值，不表示统计显著。原始逐任务日志保留六位小数；宏平均及差值由这些值计算，额外显示位数仅用于检查算术。

{macro_table}

{contrasts}

### RankMixer–KuaiRand：单组件有效，完整组合未超过仅衰减

C 和 D 相对 B 的宏平均 AUC 分别增加 0.0074025 和 0.0071000，六个任务的 AUC 和 LogLoss 均改善。E 相对 B 的宏平均 AUC 增加 0.0071855，宏平均 LogLoss 降低 0.0049285，也在全部六个任务上优于 B。

但 C 的宏平均 AUC 比 E 高 0.0002170，宏平均 LogLoss 比 E 低 0.0002745。E 相对 C 仅在 is_follow 的 AUC 上更高；E 相对 D 的宏平均 AUC 增量仅为 0.0000855。因此本组支持衰减或旋转各自能够改善内容分支，尚不支持两个组件同时使用优于仅衰减，也不能声称旋转在该组合中不可缺少。

### HyFormer–MerRec：完整组合取得最好的宏指标

C 和 D 相对 B 的宏平均 AUC 分别增加 0.0127588 和 0.0121090。E 相对 B 增加 0.0177512；相对 C、D 分别再增加 0.0049924 和 0.0056422。E 的宏平均 LogLoss 比 C、D 分别降低 0.0010166 和 0.0007852，说明在该模型—数据集组合中同时使用两个组件有额外收益。

这一优势并不覆盖每个任务的每项指标。Offer 的 E AUC 为 0.731796，低于 D 的 0.733822；E 相对 C 的 LogLoss 仅在 Like、Cart 上改善，Offer、Checkout、Purchase 分别上升 0.000031、0.000314、0.000070。D 相对 B 的 Checkout、Purchase LogLoss 也分别上升 0.000065、0.000021。不能把宏平均改善写成全部任务或概率校准均改善。

### HiFormer–TAAC-25：完整模型 AUC 最好，仅旋转 LogLoss 最低

C、D、E 相对 B 的宏平均 AUC 分别增加 0.0062000、0.0100540 和 0.0187800。E 相对 C、D 分别进一步增加 0.0125800 和 0.0087260，且点击与转化两个任务的 AUC 均高于 C、D。这支持两个组件联合使用在该组合中的排序收益。

但 D 的宏平均 LogLoss 为 0.0290100，低于 E 的 0.0302950；E−D 为 +0.0012850。E 在点击、转化上的 LogLoss 分别比 D 高 0.000301、0.002269。E 仍然优于 B、C 的 LogLoss，但没有同时实现最好的 AUC 和 LogLoss。AUC 衡量排序能力，LogLoss 对预测概率的取值也敏感；这一差异不能直接归结为校准机制，仍需额外分析。

### 交互效应的解释

描述性 AUC 交互量定义为 I = E−C−D+B。RankMixer–KuaiRand 的 I 为 −0.0073170，HyFormer–MerRec 为 −0.0071166，两组全部标签的对应交互量也为负。在 AUC 的加性尺度上，这两组完整组合的收益小于两个单独增益之和；HyFormer 的结果可称为“组合具有额外收益”，不能称为“超加性协同”。负交互量本身也不证明机制冲突或训练失败。

HiFormer–TAAC-25 的宏平均 AUC 交互量为 +0.0025260，点击、转化分别为 +0.0014170、+0.0036350。这是该单种子、该组合在 AUC 加性尺度上的正交互点估计，可作为组件互补的探索性证据；由于另外两组为负，且没有跨种子不确定性估计，不能推断稳定、普遍或统计显著的超加性协同。该组 LogLoss 的原始交互差值为 +0.0046000；LogLoss 越小越好，此数值不支持损失降低量的超加性改善。

## 5. 全部标签结果

{labels}

## 6. 参数量与控制变量的边界

下表为实际日志统计的可训练稠密参数量，冻结分支不计入。各组内骨干及 embedding 配置固定。

{parameters}

B 保留内容投影，可检查仅增加分数分支的效果，但它的有效可训练参数量并不等于 C/D/E。因此本轮是结构组件消融，不是严格等参数容量实验。可以说“衰减和旋转变体均超过仅内容分支”，不能据此完全排除参数容量差异。原始骨干 A 未在本轮重新训练，也不能用 B−A 解释新增内容分支的独立贡献。

## 7. 完成情况与协议核验

全部三组已形成正式 B/C/D/E 比较；本报告包含全部已授权的九个新增任务，无待完成训练或缺失变体。

{status_table}

九个新增任务的调度状态均为 COMPLETED、退出码均为 0:0，运行记录的返回码均为 0。每条结果都通过了唯一最终测试段、完整标签、有限指标、双日志一致、有效配置及逐 rank 数据覆盖检查。三个 E 继续使用原先固定的历史对照，不依据新增结果更换。归档包含 52 条逐标签运行记录、104 个原始 AUC/LogLoss 数值及全部指定差值。此次更新仅收集结果与完善报告，未新增或重跑训练。

## 8. 可直接用于论文的文字

### 消融实验设置

为分析 SISA 中衰减与旋转机制的作用，我们在 RankMixer–KuaiRand、HyFormer–MerRec 和 HiFormer–TAAC-25 上设计了包含四种配置的组件消融：仅内容打分（B）、内容打分与衰减（C）、内容打分与旋转（D），以及完整 SISA（E）。各配置均保留相同的骨干结构、SISA 接入位置、内容投影和每头可学习缩放，仅改变衰减与旋转的启用状态。B/C/D 均从头训练，E 复用预先指定的同协议完整模型。所有配置使用两张 H100 GPU，单卡 batch size 为 16,384，全局 batch size 为 32,768，训练一个 epoch，最大历史长度为 100。实验种子固定为 20262028，并匹配数据加载与 SISA 初始化子种子。稠密参数采用 AdamW，学习率为 10⁻⁴；稀疏参数采用 Adagrad，学习率为 0.05。模型根据验证集任务宏平均 AUC 选择，最终报告各任务及其等权宏平均 AUC 和 LogLoss。本消融沿用主实验的数据预处理、任务定义和训练预算，并以对应 H100 轮次匹配 DDP 布局及实际数据覆盖。

### 消融结果的分析

三个组合的结果表明，相较于仅内容打分，单独加入衰减或旋转均提高了全部 13 个任务的 AUC。完整 SISA 相对仅内容打分在 RankMixer–KuaiRand、HyFormer–MerRec 和 HiFormer–TAAC-25 上的宏平均 AUC 分别增加 0.0071855、0.0177512 和 0.0187800。在 HyFormer–MerRec 上，完整模型相对仅衰减和仅旋转变体的宏平均 AUC 分别增加 0.0049924 和 0.0056422，并取得最低的宏平均 LogLoss。在 HiFormer–TAAC-25 上，相应 AUC 增量为 0.0125800 和 0.0087260，但仅旋转变体取得最低 LogLoss。在 RankMixer–KuaiRand 上，仅衰减变体同时取得最高 AUC 和最低 LogLoss。结果支持衰减与旋转各自的增益，以及在两个组合中联合使用带来的额外 AUC 收益，同时表明最优组件配置依赖架构、数据与评价指标。HiFormer–TAAC-25 呈现正 AUC 交互点估计，另外两组为负；这些单种子结果不足以主张普遍协同、两组件普遍不可替代或统计显著性。

## 9. 证据边界与使用说明

这份报告包含已授权三组消融的完整最终结果，可用于论文的实验设置、三组结果表及分析。当前没有多种子消融，无法估计增益的跨种子方差；三组来自已知有效场景，无法外推至全部主实验组合；没有固定衰减或固定旋转对照，无法单独证明内容自适应优于固定位置规则；没有受控的延迟、显存或吞吐测量，不能由任务时长推断效率优势。

正式结果经 checkpoint 最终测试段、对应 Slurm 输出、完成标记、有效配置及逐 rank 数据覆盖交叉核验。九个新增任务均成功退出，最终指标在两份日志中一致；历史 E 同时核对原始完成标记与既有归档。采集程序只纳入通过这些检查的运行。完整运行映射、逐任务 CSV、差值 CSV 和独立审计说明随报告归档；前两组的旧审计注明历史快照，第三组另有最终审计。原始训练日志及模型文件保留在运行目录，不作为论文附件提交。
"""
    assert "http://" not in content and "https://" not in content
    (REPORT_DIR / "report.md").write_text(content, encoding="utf-8")
    print(f"Wrote {REPORT_DIR / 'report.md'} ({len(content.splitlines())} lines)")


if __name__ == "__main__":
    main()
