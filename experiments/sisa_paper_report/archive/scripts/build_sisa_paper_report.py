#!/usr/bin/env python3
"""Build a complete standalone Chinese paper experiment report without links."""

from __future__ import annotations

import csv
import json
import re
import statistics as stats
from pathlib import Path

from audit_sisa_paper_evidence import LABELS, MODELS, NAMES, SEEDS, read_csv


ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file() and (parent / "unirank").is_dir()
)
STUDY = ROOT / "experiments/sisa_paper_report"
RESULTS = STUDY / "archive/results"


def main():
    cells = read_csv(RESULTS / "cell_results.csv")
    labels = read_csv(RESULTS / "label_results.csv")
    runs = read_csv(RESULTS / "run_evidence.csv")
    counts = read_csv(RESULTS / "dataset_counts.csv")
    resources = read_csv(RESULTS / "resource_usage.csv")
    segments = read_csv(RESULTS / "segment_results.csv")
    summary = json.loads((RESULTS / "summary.json").read_text())
    f = lambda value: f"{float(value):.6f}"
    sign = lambda value: f"{float(value):+.6f}"
    model_dataset = lambda r: [r["model"], NAMES[r["dataset"]]]
    sources = [
        {"id": "audit", "label": "从正式任务 CSV 和最终测试日志独立复算的论文证据", "path": "experiments/sisa_paper_report/archive/scripts/audit_sisa_paper_evidence.py",
         "query": {"engine": "python", "language": "python", "description": "固定 accepted 4×4 范围，核对 64 次正式运行，再先标签等权、后单元等权。",
                   "sql": "import runpy\nrunpy.run_path('experiments/sisa_paper_report/archive/scripts/audit_sisa_paper_evidence.py', run_name='__main__')\n# The report builder reads these audited CSVs; chart deltas are multiplied by 1000 for display.\n",
                   "tables_used": ["experiments/sisa_paper_report/archive/results/"+name for name in ("label_results.csv", "cell_results.csv", "summary.json", "run_evidence.csv", "coverage_audit.csv", "segment_results.csv", "resource_usage.csv", "dataset_metadata.json", "source_manifest.csv")],
                   "filters": ["HiFormer, HyFormer, RankMixer, Zenith", "QK-Video, KuaiRand, TAAC-25, MerRec", "Final selected runs only; exclude diagnostic scope"],
                   "metric_definitions": ["cell macro = mean over 16 cells of within-cell label means", "delta = SISA minus baseline seed20262027", "three-seed SD is descriptive sample SD of cell-level means, not paired-effect uncertainty"]}},
        {"id": "method", "label": "SISA 实现、原生接入点与实际日志参数", "path": "unirank/pytorch/layers/attentions/sisa.py",
         "query": {"engine": "python", "language": "python", "sql": "from pathlib import Path\nsource_text = Path('unirank/pytorch/layers/attentions/sisa.py').read_text()\n", "description": "实现文本与日志参数交叉核对；当前工作区已有语法错误，不能视为通过运行验证。", "tables_used": ["model_zoo/HiFormer.py", "model_zoo/HyFormer.py", "model_zoo/RankMixer.py", "model_zoo/Zenith.py", "unirank/pytorch/models/multitask_model.py", "experiments/sisa_paper_report/archive/results/logged_configurations.json"]}},
        {"id": "data", "label": "数据元信息、Parquet footer 与各 rank 加载记录", "path": "experiments/sisa_paper_report/archive/results/dataset_metadata.json",
         "query": {"engine": "python", "language": "python", "sql": "import json\nfrom pathlib import Path\nrows = json.loads(Path('experiments/sisa_paper_report/archive/results/dataset_metadata.json').read_text())\n", "description": "只读取 footer 行数，不重新扫描完整样本；历史划分日期以元信息记录为准。", "tables_used": ["experiments/sisa_paper_report/archive/results/dataset_counts.csv", "experiments/sisa_paper_report/archive/results/coverage_audit.csv"]}},
        {"id": "paper", "label": "仓库内论文表值转录，四位小数", "path": "experiments/sisa_three_seed_unified/sources/paper_table.csv",
         "query": {"description": "来源说明将其标为 UniRank arXiv:2607.19987 Table 2，2026-08-26 下载版本。本次未联网重新核验原文；仅作为次级参考。", "tables_used": ["experiments/sisa_three_seed_unified/sources/README.md"]}},
    ]
    blocks, tables, charts, datasets, md = [], [], [], {}, []

    def prose(id_, body, source="audit"):
        block = {"id": id_, "type": "markdown", "body": body.strip()}
        if source:
            block["sourceId"] = source
        blocks.append(block)
        md.append(body.strip())

    def table(id_, title, headers, values, source="audit"):
        fields = [f"c{i}" for i in range(len(headers))]
        data = [dict(zip(fields, map(str, row))) for row in values]
        datasets[id_] = data
        tables.append({"id": id_, "title": title, "dataset": id_, "sourceId": source,
                       "layout": "full", "density": "comfortable", "defaultSort": {"field": "c0", "direction": "asc"},
                       "columns": [{"field": field, "label": name, "type": "text"} for field, name in zip(fields, headers)]})
        blocks.append({"id": id_+"_block", "type": "table", "tableId": id_})
        md.append("**"+title+"**\n\n| "+" | ".join(headers)+" |\n| "+" | ".join("---" for _ in headers)+" |\n"+"\n".join("| "+" | ".join(str(x).replace("|", "/") for x in row)+" |" for row in values))

    def chart(id_, title, values, question, interpretation):
        datasets[id_] = values
        charts.append({"id": id_, "title": title, "subtitle": "单位：10⁻³ AUC；零点为无变化", "type": "horizontalBar", "intent": "comparison", "question": question,
                       "rationale": "16 个固定单元的正负差值，需要逐一比较方向与幅度；采用含零点的横向条形图。",
                       "dataset": id_, "sourceId": "audit", "palette": {"kind": "diverging", "midpoint": 0}, "labels": {"values": "auto"},
                       "encodings": {"x": {"field": "unit", "type": "nominal", "aggregate": "none", "label": "模型 × 数据集"},
                                     "y": {"field": "delta_milli", "type": "quantitative", "aggregate": "none", "label": "ΔAUC × 10⁻³"},
                                     "tooltip": [{"field": "baseline", "type": "quantitative", "label": "比较起点 AUC"}, {"field": "sisa", "type": "quantitative", "label": "比较终点 AUC"}, {"field": "labels", "type": "quantitative", "label": "标签数"}]},
                       "xAxisTitle": "模型 × 数据集", "yAxisTitle": "ΔAUC × 10⁻³", "valueFormat": "number", "layout": "full", "maxRows": 16})
        blocks.append({"id": id_+"_block", "type": "chart", "chartId": id_})
        # The user selected a standalone copyable text report; tables carry
        # every quantitative value without requiring an external chart reader.

    title = "UniRank SISA 实验设置与结果报告"
    prose("title", "# "+title, None)
    prose("summary", """## 技术摘要：单 seed 配对支持整体改善，多 seed 因果结论仍待补齐

本报告整理 UniRank 中 SISA 的论文实验材料，证据冻结日为 **2026-09-14（Asia/Shanghai）**，实验运行集中于 2026 年 8 月。正式范围限定为 **HiFormer、HyFormer、RankMixer、Zenith × QK-Video、KuaiRand、TAAC-25、MerRec**，共 16 个模型–数据集单元、17 种数据集–标签任务和 68 个模型–数据集–标签组合。

**建议主文采用 seed 20262027 的本地 baseline／SISA 配对。** 每个单元先对标签等权，再对 16 个单元等权，AUC 平均差值为 **+0.005841**（绝对 AUC，即 +0.5841 个百分点），15/16 单元和 57/68 标签组合为正；Logloss 宏平均差值为 **−0.002383**，16/16 单元、55/68 标签组合改善。这是一轮相同协议下的配对点估计，没有置信区间或显著性检验。

**三轮 SISA 均值可作补充描述。** seeds 20262027／20262028／20262029 相对单轮 baseline 的单元宏平均 ΔAUC 为 +0.006415，15/16 单元为正；但后两轮从 4×8192 改为 2×16384，改变了 blocked loader 的 train／valid／test 尾批样本集合。因而三点均值与样本标准差不能解释为三个同协议 baseline／SISA 配对的效果或纯随机种子方差。

收益主要来自 TAAC-25、KuaiRand 和 MerRec；QK-Video 的配对宏平均仅 +0.000218，HiFormer–QK-Video 为明确负面结果。当前可以写“在已测架构和数据集上通常改善离线预测”，尚不支持“所有任务提升”“统计显著”“在线业务收益”或“已验证低时延开销”。""")
    prose("scope", """## 1. 证据范围与研究问题

本报告回答三个已具备数据的问题：SISA 接入原有注意力位置后是否改善离线预测；收益是否因模型、数据集和动作标签不同而变化；现有复跑是否显示明显不稳定性。衰减、旋转、可学习头权重各自贡献多少，仍需受控消融回答。

正式证据由 strict 档案中的 9 个单元与 expansion 档案中的 7 个单元组成，并加入两轮完整 SISA 复跑：**16 baseline + 48 SISA = 64 次最终选定运行**。每次运行包含多个动作任务；共 272 组 AUC／Logloss，即 544 个标量。一次失败后的重试、DDP rank seed、dataloader seed 和参数初始化 seed 均不另算独立实验。

OneTrans、UniMixer、UltraHSTU、Taobao 及其他配置的记录保留为历史诊断档案，不进入本报告的平均值、极值、图表、结论或后续实验矩阵。旧 strict／expansion 报告中的全量总数不能直接挪作当前论文结论。""")
    table("inventory", "表 1：正式证据组成", ["来源", "范围", "最终运行", "用途"], [
        ["sisa_native_strict", "HiFormer / RankMixer / Zenith × QK / KuaiRand / MerRec", "18 = 9 对", "主实验的一部分"],
        ["sisa_expansion_acd", "HyFormer × 4 数据集；另 3 模型 × TAAC-25", "14 = 7 对", "补齐主实验矩阵"],
        ["sisa_single_seed20262028", "4 × 4，SISA", "16", "第二轮描述性结果"],
        ["sisa_single_seed20262029", "4 × 4，SISA", "16", "第三轮；含 task 0 最终重试"],
    ])
    prose("data", """## 2. 数据规模、标签与预处理口径

四个数据集均为多任务二分类，使用仓库已预处理的 Parquet 分块、用户侧表和物品侧表。下表是**预处理后、drop_last 前的交互样本数**，不是原始下载量、独立用户数或实际评测样本数。本次用全部 12 个划分的 Parquet footer、meta_data.json 和历史各 rank 分配记录交叉核对，总行数一致。

模型均将历史最大长度设为 100；加载器按 seq_len 截取历史，并与目标物品组合。TAAC-25 的列表型静态特征另有自身长度设置，不能把其长度 5 与历史长度 100 混用。词表大小含 padding／OOV 编码，不能直接作为精确独立用户、物品数。
""", "data")
    table("data_counts", "表 2：预处理后的样本数", ["数据集", "训练", "验证", "测试", "标签数"], [[r["dataset_display"], *(f"{int(r[f'{s}_assigned']):,}" for s in ("train", "valid", "test")), r["labels"]] for r in counts], "data")
    table("label_definitions", "表 3：完整动作标签映射", ["展示名", "配置 ID", "标签（保留大小写）"], [[NAMES[d], d, ", ".join(names)] for d,names in LABELS.items()], "data")
    prose("preprocessing", """### 划分与样本构造

- **QK-Video**：预处理实现按每个用户已有序列顺序切分，默认目标比例 0.8／0.1／0.1。小用户序列的整数分配使实际比例不必精确等于该比例；不应描述为随机交互切分，也未核验为一个统一全局日期边界。
- **KuaiRand**：元信息记录训练为 2022-04-07 至 2022-05-02，验证为 05-03 至 05-05，测试为 05-06 至 05-08。
- **TAAC-25**：元信息记录训练为 2024-10-15 至 2025-05-28，验证为 2025-05-29，测试为 05-30 至 08-05；时区 Asia/Shanghai。末尾划分受正样本数约束，并过滤无 click／conversion 的用户序列，最少用户交互数设为 10。这些规则定义了评估人群，不能外推为未筛选全体流量。验证、测试的时间跨度及标签分布明显不同。
- **MerRec**：训练为 2023-10-01 至 10-25，验证为 10-26 至 10-28，测试为 10-29 至 10-31。每个 user–item 的首次 item_view 是样本锚点，其后 Like／Cart／Offer／Checkout／Purchase 行为按 OR 合成为多标签；非 view 行为用于填标签，不保留为样本或序列 token。投稿时应交代标签归因窗口以及是否跨越划分边界。

元信息均记录低频特征阈值 2，但适用字段不同。上述内容是当前元信息和实现所记录的协议；本次未重跑预处理，也未进行逐事件时序、未来标签泄漏或全量内容哈希审计。**相同总行数本身不能证明样本内容完全相同或不存在泄漏。**

MerRec 的 meta_data.json 声称 train／valid／test 为 32／16／16 块，但当前物理文件与训练日志均为 **8／8／8** 块；报告采用实际日志和文件口径，并保留元信息过时的审计说明。""", "data")
    prose("metrics", """## 3. 指标和汇总定义

每个标签单独计算测试 **ROC-AUC（越高越好）**与**二元 Logloss（越低越好）**。实现将各 rank 的预测和标签汇集后计算指标，不对 rank AUC 简单求均值。本报告的 AUC 不是 gAUC／UAUC、NDCG 或线上 CTR。

令 c=(模型, 数据集)，L_c 为该单元标签数，A(c,l,s) 为第 s 轮 SISA 的标签 AUC，B(c,l) 为 seed 20262027 baseline：

- 单元 AUC：A_c(s) = Σ_l A(c,l,s) / L_c。
- 配对效果：Δ_c = A_c(20262027) − Σ_l B(c,l) / L_c。
- **主指标 cell-macro**：Δ_macro = Σ_c Δ_c / 16。模型汇总对四个数据集等权；数据集汇总对四个模型等权。
- **标签等权补充指标 label-macro**：68 个标签组合的差值直接等权。此时 KuaiRand 权重 24/68，TAAC-25 权重 8/68，故不与 cell-macro 混称同一个“平均提升”。
- 三轮均值：对三个实验 seed 的单元 AUC 求均值；样本 SD 的分母为 3−1。先算单元 AUC 再算 SD，不能平均标签 SD 来替代。

Logloss 使用相同的汇总顺序。所有 Δ 都为 **SISA − baseline**；数表保留六位小数，仅代表来自六位小数日志的重算精度。+0.005841 AUC = +0.5841 个百分点，不是 +0.5841% 相对增长。不同标签的基准发生率不同，宏平均 Logloss 不代表一个合并总体的损失。""")
    prose("method", """## 4. 模型设定与 SISA 接入方式

SISA-v2 在原有 attention 的 softmax 前加入分数偏置，保持基础 Q／K／V、残差、前馈网络及硬掩码路径。可用下式描述其接入位置：

`score_h(i,j) = q_h(i)ᵀ k_h(j) / sqrt(d_h) + bias_h(i,j) + mask(i,j)`。

实现为每头构建 b／c 投影、输入相关衰减和成对旋转通道。对有效位置，`log α_t = −softplus(w_α x_t + b_α)`；分别对 log α 和相位做前缀和，旋转投影后的 b／c 通道，并以指数衰减缩放。query 模式使用查询位置累计衰减，sequence_end 模式使用序列末端累计衰减；同时以中点居中、指数截断控制数值范围。最终通道乘积形成偏置，并乘以正的可学习每头权重 `λ_h = softplus(raw_λ_h) + 1e−8`。具体通道顺序、截断和 mask 以实现为准，不应把无截断的理论形式当成所有数值区间下的精确运行公式。

独立 CPU 随机数生成器初始化新增参数，目的是不消耗 backbone 初始化所用的全局 RNG；不同接入位置使用子流偏移。默认参数为 score_dim=16、lambda_init=0.1、decay_bias=−5、exp_clamp=11、score_scale=1。前两个及 scale 已由正式运行日志直接核对；旧日志未显式写出的默认值来自实现／历史协议，不伪称为逐运行显式记录。
""", "method")
    table("sites", "表 4：原生接入位置", ["模型", "SISA 接入", "解释边界"], [
        ["HiFormer", "target attention；原生层内 attention", "query / sequence_end 分别匹配原位置"],
        ["HyFormer", "序列 self-attention；query-to-sequence attention", "同时涉及多类位置，尚无分位置消融"],
        ["RankMixer", "既有 target-attention pooling", "原 token mixer 未替换；不能称 SISA 改善 mixer 内注意力"],
        ["Zenith", "target attention；原生 token attention", "原有 mask 约束继续生效"],
    ], "method")
    prose("training", """## 5. 训练、随机性和评测协议

主实验每个 baseline／SISA 配对使用相同数据路径、预处理特征、epoch、global batch、种子语义和评测流程。GPU 型号不要求跨所有单元完全相同，但记录每个单元的硬件；主实验内 baseline／SISA 配对硬件可从运行表核查。

共同设置为：1 epoch，历史长度 100，embedding dimension 16，3 层，token dimension 256，任务 tower 为 [128,64]，ReLU，配置中的 net dropout 为 0；稠密 AdamW 学习率 1e−4、weight decay 0，稀疏 Adagrad 学习率 0.05。各动作 BCE 先按样本取均值，再以相等权重求和。DDP blocked 路径在 epoch 末同步验证，监控验证集标签宏平均 AUC 并加载最佳 checkpoint 做一次正式测试；此次只有一个 epoch 的验证候选，不是按测试 AUC 选 checkpoint。

HiFormer 使用 2 heads、qkv_rank=256；HyFormer 使用 2 heads、4 个非序列 token、2 个 global token；RankMixer 使用 Chunk tokenizer、4 个非序列 token；Zenith 使用 id_dim=48、2 个 token。接入点内部默认 head 设置不等同于所有模型共享显式 num_heads=2。
""", "method")
    table("protocol", "表 5：实际实验协议", ["系列", "实验 seed", "GPU / 每卡 batch", "global batch", "内部 RNG", "可比性"], [
        ["Baseline 与 SISA 主配对", "20262027", "4 卡；8192；RTX4090/L40S/H100 按单元记录", "32768", "loader=2026；SISA parameter=20260821 为历史默认", "同单元配对"],
        ["SISA 第二轮", "20262028", "2×H100 80GB；16384", "32768", "loader=2027；parameter=20260822", "与第三轮同协议"],
        ["SISA 第三轮", "20262029", "2×H100 80GB；16384", "32768", "loader=2028；parameter=20260823", "缺少匹配 baseline"],
    ])
    table("common_settings", "实验设置汇总：可直接用于论文 Implementation Details", ["设置项", "实际设置", "说明"], [
        ["任务", "多任务二分类", "每个数据集同时预测其全部动作标签"],
        ["最大历史序列长度", "100", "按用户历史长度截取，不包含目标物品位置"],
        ["Embedding dimension", "16", "基础特征嵌入维度；Zenith 另设 id_dim=48"],
        ["主干层数 / token dimension", "3 / 256", "四个 backbone 的基础配置"],
        ["任务预测塔", "[128, 64]，ReLU", "每个动作一个预测输出"],
        ["训练损失", "各任务 binary cross-entropy 的等权和", "各任务先对 batch 样本求均值"],
        ["稠密优化器", "AdamW，学习率 0.0001", "dense weight decay=0"],
        ["稀疏优化器", "Adagrad，学习率 0.05", "Zenith–MerRec 存在等公式分块实现"],
        ["训练轮数", "1 epoch", "不是训练至收敛后的多 epoch 比较"],
        ["梯度累积", "1", "不额外累积多个 batch"],
        ["主配对 batch", "每卡 8192 × 4 卡", "配置 global batch=32768"],
        ["后两轮 SISA batch", "每卡 16384 × 2 卡", "配置 global batch=32768；尾批覆盖不同"],
        ["训练精度", "BF16", "所有正式运行日志均启用"],
        ["编译 / 梯度检查点", "torch.compile 启用 / gradient checkpointing 关闭", "以实际运行合并配置为准"],
        ["Dropout", "配置中的 net_dropout=0", "HiFormer/RankMixer/Zenith 显式 attention_dropout=0"],
        ["验证时机", "每个 epoch 末同步验证", "DDP blocked 路径关闭 step-wise 验证"],
        ["模型选择", "验证集各标签 AUC 的宏平均最大", "不按测试集表现选 checkpoint"],
        ["测试", "加载最佳 checkpoint 后评估", "各 rank 汇集预测后逐标签计算 AUC/Logloss"],
        ["尾批", "每个 Parquet 块内 drop_last", "train/valid/test 均受影响"],
        ["SISA 参数", "score_dim=16；lambda_init=0.1；score_scale=1", "主配对与两轮补充实验一致"],
        ["SISA 数值默认值", "decay_bias=−5；exp_clamp=11", "来自实现与历史协议，旧日志未逐项显式记录"],
    ])
    hardware_names = {"l40s": "4×L40S", "rtx4090": "4×RTX4090", "NVIDIA H100 80GB HBM3": "4×H100 80GB"}
    table("hardware_matrix", "主配对硬件：每个单元的 baseline 与 SISA 使用相同 GPU 类型", ["模型", "QK-Video", "KuaiRand", "TAAC-25", "MerRec"], [[model]+[hardware_names[next(r['gpu'] for r in runs if r['model']==model and r['dataset']==d and r['setting']=='baseline')] for d in LABELS] for model in MODELS])
    prose("software_environment", """### 软件环境与实现版本

项目使用 uv 管理环境，Python 3.12，PyTorch 2.8.0，CUDA 12.6 对应的 PyTorch 构建。历史严格矩阵报告记录 Python 3.12.12、PyTorch 2.8.0+cu126、CUDA runtime 12.6、cuDNN 9.10.2 和 PyArrow 25.0.1。上述精确补丁版本是该历史矩阵记录，不能未经逐运行核对就扩展为所有集群任务完全相同的软件栈。

同一配对保持模型、数据与预算一致；不同单元之间的 GPU 类型如上表记录。后两轮 SISA 全部使用 2×H100 80GB，不要求与旧主配对具有相同 GPU 型号，但必须承认 DDP 布局和评测尾批改变所带来的可比性限制。""")
    prose("rng", """各轮 accumulation_steps=1，BF16。DDP rank 的 effective seed = base seed + rank；rank 随机流、loader 子流、SISA 参数子流属于同一轮实验。后两轮未使用四卡 OOM fallback。

Zenith–MerRec 主配对的最终 baseline 使用 14843_30，SISA 使用 14846_31；稀疏 Adagrad 的临时张量分块是为完成训练的内存兼容处理。日志中该单元除 SISA 开关参数外还记录 sparse_adagrad_chunk_size=16777216 的差异；历史报告和对应测试记录其更新公式等价，但不能据此宣称所有底层执行路径逐位相同。

seed 20262029 的 task 0 最终选 549797_0（attempt2），而不是训练后 shell 出错的 549595_0。HyFormer–QK baseline 的 543255_18 Slurm 日志包含两段训练会话，本次选择与归档正式指标对应的最后测试段，重复会话不另计 seed。精确任务映射和原始日志路径见 `results/run_evidence.csv`。""")
    prose("main_auc", """## 6. 主实验：15/16 单元 AUC 改善，全部单元 Logloss 改善

表 6 是论文主表的建议版本，baseline 和 SISA 均来自 seed 20262027；所有表项都是单元内部标签等权平均。平均 ΔAUC 为 **+0.005841**，中位数 +0.006700；15/16 单元正向。标签等权平均 ΔAUC 为 +0.005356，57/68 标签组合正向。两种加权均为正，说明整体方向不是由一种汇总权重单独造成。

四个模型各有适用与收益幅度差异。该表支持已测试范围内的离线效果，不包含抽样误差、baseline 种子方差、显著性星号或跨域泛化保证。""")
    table("main_auc_table", "表 6：单 seed 同协议配对 AUC（↑）", ["模型", "数据集", "Baseline", "+SISA", "ΔAUC", "正向标签"], [model_dataset(r)+[f(r["baseline_auc"]), f(r["sisa_20262027_auc"]), sign(r["paired_delta_auc"]), f"{r['paired_positive_auc_labels']}/{r['labels']}"] for r in cells])
    chart("paired_gain", "主配对的单元 AUC 差值", [{"unit": " / ".join(model_dataset(r)), "delta_milli": float(r["paired_delta_auc"])*1000, "baseline": float(r["baseline_auc"]), "sisa": float(r["sisa_20262027_auc"]), "labels": int(r["labels"])} for r in sorted(cells,key=lambda r:float(r["paired_delta_auc"]),reverse=True)], "哪些单元贡献主实验的提升，哪些出现退化？", "HiFormer–QK-Video 为负，其余 15 个单元为正。")
    prose("main_loss", """### Logloss 的方向更一致，但并非全部标签同时改善

单元宏平均 ΔLogloss 为 **−0.002383**，16/16 单元下降；标签等权为 −0.002561，55/68 标签组合下降。HiFormer–QK-Video 虽然 AUC 降低，Logloss 仍小幅下降 −0.000029，两项指标不能互相替代。AUC 衡量排序区分能力，Logloss 对预测概率也敏感；这里未测 ECE 等校准指标，不能把 Logloss 下降单独写成“已证实改善概率校准”。""")
    table("main_loss_table", "表 7：单 seed 同协议配对 Logloss（↓）", ["模型", "数据集", "Baseline", "+SISA", "ΔLogloss"], [model_dataset(r)+[f(r["baseline_logloss"]), f(r["sisa_20262027_logloss"]), sign(r["paired_delta_logloss"])] for r in cells])
    prose("segments", """## 7. 收益的异质性：TAAC-25 最强，QK-Video 接近中性

主实验中 TAAC-25 的四模型平均 ΔAUC 为 **+0.010656**，KuaiRand 为 +0.006420，MerRec 为 +0.006069，QK-Video 仅 +0.000218。主文应保留该差异，避免用一个整体数字暗示全部数据集同等受益。

按模型，HyFormer 的四数据集宏平均最高（+0.007824），其次为 RankMixer（+0.006899）、Zenith（+0.005056）和 HiFormer（+0.003585）。这衡量的是各自 baseline 的改善幅度，不等于最终绝对 AUC 排名；模型间结构、参数量和 baseline 水平本来不同。
""")
    table("segments_table", "表 8：主配对及三轮描述性汇总", ["维度", "分组", "配对 ΔAUC", "配对 ΔLogloss", "三轮均值 ΔAUC"], [["模型" if r["dimension"]=="model" else "数据集", NAMES.get(r["value"],r["value"]), sign(r["paired_delta_auc"]), sign(r["paired_delta_logloss"]), sign(r["three_seed_delta_auc"])] for r in segments])
    prose("actions", """### 动作级结果提示：单元改善不能解释为所有业务目标都改善

KuaiRand 的 click 和 long_view 在四模型上均正向，其动作宏平均 ΔAUC 分别为 +0.014351、+0.011150；TAAC-25 的 click／conversion 均在四模型上提升。MerRec 的 Like／Cart 平均改善较大，而 Offer 动作宏平均为 −0.002616；Purchase 只有 1/4 模型提升，其动作平均仍为 +0.000713，说明平均值会掩盖多数模型的回退。

最大的标签级收益是 **HyFormer–MerRec Like：+0.023690**；最大退化是 **HiFormer–MerRec Offer：−0.013953**。HiFormer–QK 的单元 AUC 降低并非可忽略的显示精度问题；其中 follow 为 −0.005990，三轮 SISA 单元均值也都低于旧 baseline。当前没有注意力可视化、用户切片或受控消融证据，不能进一步断言退化由某一 SISA 组件、序列长度或类别稀疏度造成。""")
    action_rows=[]
    for d, names in LABELS.items():
        for name in names:
            group=[r for r in labels if r["dataset"]==d and r["label"]==name]
            action_rows.append([NAMES[d], name, sign(stats.fmean(float(r["paired_delta_auc"]) for r in group)), f"{sum(float(r['paired_delta_auc'])>0 for r in group)}/4"])
    table("actions_table", "表 9：动作维度的四模型平均配对 ΔAUC", ["数据集", "动作", "ΔAUC", "正向模型"], action_rows)
    prose("three_seeds", """## 8. 三轮 SISA：均值改善可描述，不能作为三轮配对效应

三轮 SISA 的单元宏平均 ΔAUC（对比固定 seed 20262027 baseline）依次为 **+0.005841、+0.007460、+0.005944**，正向单元依次为 15／15／14。三轮均值的宏平均 ΔAUC 为 +0.006415，标签等权为 +0.005537；54/68 标签组合的均值更高，39/68 在每轮都高于旧 baseline。14/16 单元在每轮均为正，但这仍然是对同一个单轮对照的重复比较。

三轮均值的 Logloss 宏平均 Δ 为 −0.001513，14/16 单元更低；退化单元为 HiFormer–QK 和 RankMixer–TAAC-25。逐轮 Logloss 正向单元为 16／13／10，表明较好的平均 AUC 不能取代对概率损失稳定性的检查。

下表 SD 是三个**单元 AUC 均值**的样本标准差，包括随机流、硬件／DDP 划分与尾批评测集合差异的共同影响；不能标为 ΔAUC 的标准误、95% CI 或纯 seed 方差。""")
    table("seed_table", "表 10：三轮 SISA 的描述性 AUC", ["模型", "数据集", "20262027", "20262028", "20262029", "Mean", "SD", "Mean−Baseline"], [model_dataset(r)+[*(f(r[f"sisa_{seed}_auc"]) for seed in SEEDS), f(r["three_seed_mean_auc"]), f(r["three_seed_sd_auc"]), sign(r["three_seed_delta_auc"])] for r in cells])
    prose("coverage", """### Global batch 相同，实际评测集合仍有变化

blocked loader 在每个文件块内丢弃不满 batch 的尾批。4×8192 与 2×16384 虽然 nominal global batch 都为 32768，但各 rank 分块和每块 batch 边界改变，训练、验证、测试实际处理行数均改变。各 rank batch 数还可能不相等，DDP join 处理末尾不均衡；表中的 global batch 应理解为配置值，不保证最后所有更新都有四卡或两卡满贡献。

测试覆盖变化最大的是 MerRec：旧协议处理 11,640,832／11,681,154（99.654812%），新协议处理 11,599,872／11,681,154（99.304161%），额外丢弃 40,960 行，覆盖率降低 **0.350650 个百分点**。样本占比小也不能证明对稀有标签 AUC 没有影响。
""")
    table("test_coverage_table", "表 11：测试集处理行数与覆盖率", ["数据集", "分配样本数", "旧协议处理", "新协议处理", "旧覆盖率", "新覆盖率"], [[r["dataset_display"], f"{int(r['test_assigned']):,}", f"{int(r['test_ws4_processed']):,}", f"{int(r['test_ws2_processed']):,}", f"{int(r['test_ws4_processed'])/int(r['test_assigned'])*100:.6f}%", f"{int(r['test_ws2_processed'])/int(r['test_assigned'])*100:.6f}%"] for r in counts])
    prose("seed_pair", """### 后两轮同协议复跑仍有单元波动

20262028 与 20262029 的每个数据集各 rank train／valid／test 行数和步数一致，适合描述同协议 SISA 重复运行的变化。两轮间单元 AUC 的平均绝对差为 **0.002989**。较大差异包括 Zenith–TAAC-25（20262029−20262028 = −0.009036）、HyFormer–TAAC-25（−0.007010）、Zenith–MerRec（−0.006175）和 HyFormer–MerRec（+0.005710）。这两轮仍只有 SISA，没有相同 seed 的 baseline，因此不能估计 SISA 增益本身的 seed 波动。
""")
    chart("same_protocol_change", "同协议 SISA：20262029 减 20262028", [{"unit": " / ".join(model_dataset(r)), "delta_milli": (float(r["sisa_20262029_auc"])-float(r["sisa_20262028_auc"]))*1000, "baseline": float(r["sisa_20262028_auc"]), "sisa": float(r["sisa_20262029_auc"]), "labels": int(r["labels"])} for r in sorted(cells,key=lambda r:float(r["sisa_20262029_auc"])-float(r["sisa_20262028_auc"]),reverse=True)], "同协议 SISA 两轮复跑的单元均值变化多大？", "横轴变化反映两轮 SISA 之差，不是相对无 SISA baseline 的效果。")
    prose("baseline", """## 9. Baseline 复现及论文表值参考

**效果归因优先使用本地配对 baseline。** 论文发布表值只能提供复现背景，不能替代同数据、同训练预算的控制组。按仓库转录的四位小数论文 AUC，比对 68 个匹配标签，本地 baseline 平均绝对差 **0.001458**，最大绝对差 **0.008859**，68/68 在 0.01 内。这只是在给定阈值内的数值接近，不构成数据预处理、环境、概率校准或完整论文设置均一致的证明。

三轮 SISA 对论文表值的 cell-macro ΔAUC 为 +0.006817，仅适合作次级描述。论文四位小数和日志六位小数精度不同，不据此作极小差值的名次或显著性判断。尤其不能用不同来源混合挑选每个标签“最有利的 baseline”。

仓库来源说明把该转录标为 UniRank arXiv:2607.19987 的 Table 2（Sequence Length=100），下载版本日期 2026-08-26。本次审计使用这份本地转录，**未联网重新核验论文原文、版本或表号**。投稿前应逐项校验正式引文和原表；早期实验档案存在不同表号表述，不能直接照抄。""")
    baseline_rows=[]
    for r in cells:
        group=[x for x in labels if x['model']==r['model'] and x['dataset']==r['dataset']]
        baseline_rows.append(model_dataset(r)+[f(stats.fmean(abs(float(x['baseline_auc'])-float(x['paper_auc'])) for x in group)), f(max(abs(float(x['baseline_auc'])-float(x['paper_auc'])) for x in group))])
    table("baseline_table", "表 12：本地 baseline 与仓库论文转录 AUC 的差异", ["模型", "数据集", "标签 MAE", "最大绝对差"], baseline_rows)
    prose("efficiency", """## 10. 参数与资源：新增参数少，端到端效率尚未受控测量

正式日志可直接核对 baseline／SISA 参数量差值。各接入模块的新增参数为 `H × (2.5 × score_dim × D + D + 2)`，其中 D 是接入位置的输入维度、H 是该位置头数；整模型对各位置求和。该表达式来自新增 b、c、phase、decay、decay bias 和 raw head scale 参数形状。

所有单元新增总参数比例低于 0.086%；其中 RankMixer 仅在 pooling 位置接入，因此参数最少。但总参数分母包含巨大的 embedding 表，不能用这个比例暗示计算或 activation 显存开销同样小。对稠密参数的比例更有解释力，见下表。""", "method")
    parameter_rows=[]
    for model in MODELS:
        values=[]
        for a in runs:
            if a['model']==model and a['seed']=='20262027' and a['setting']=='baseline':
                b=next(x for x in runs if x['model']==model and x['dataset']==a['dataset'] and x['seed']=='20262027' and x['setting']=='sisa')
                added=int(b['total_parameters'])-int(a['total_parameters'])
                values.append((added,100*added/int(a['total_parameters']),100*added/int(a['dense_parameters'])))
        parameter_rows.append([model, f"{min(x[0] for x in values):,}–{max(x[0] for x in values):,}", f"{min(x[1] for x in values):.4f}%–{max(x[1] for x in values):.4f}%", f"{min(x[2] for x in values):.3f}%–{max(x[2] for x in values):.3f}%"])
    table("parameter_table", "表 13：四数据集中的新增参数范围", ["模型", "新增参数", "占 baseline 总参数", "占 baseline 稠密参数"], parameter_rows)
    prose("runtime", """后两轮每次使用 2×H100。按照归档 Slurm elapsed×2 计算，两轮 16 个正式任务合计分别约 **33.148、32.859 GPU·小时**；这包含所选任务的进程启动、编译、训练和评测时间，不含排队和未选失败尝试，也不是项目全部探索成本。

归档的 30 秒采样显存最大值分别为 71,847、71,842 MiB；这里是 sampled per-device memory usage，可能漏掉瞬时峰值，且不是独立的模型 activation 开销。缺少相同硬件、batch、编译状态下 baseline／SISA 的重复计时、吞吐和推理时延对照，因此**不能从两卡比四卡的 elapsed 变化推断 SISA 加速，也不能写“零开销”**。""")
    def elapsed(v):
        seconds=int(v);return f"{seconds//3600:02}:{seconds%3600//60:02}:{seconds%60:02}"
    rr=[]
    for r in cells:
        a,b=[next(x for x in resources if x['model']==r['model'] and x['dataset']==r['dataset'] and x['seed']==str(s)) for s in (20262028,20262029)]
        rr.append(model_dataset(r)+[elapsed(a['elapsed_seconds']),elapsed(b['elapsed_seconds']),f"{max(int(a['sampled_peak_memory_mib']),int(b['sampled_peak_memory_mib'])):,}"])
    table("runtime_table", "表 14：两卡 H100 正式 SISA 运行资源记录", ["模型", "数据集", "20262028 elapsed", "20262029 elapsed", "两轮最大采样 MiB/卡"], rr)
    prose("validation", """## 11. 证据验证与稳健性边界

本次审计从原始 checkpoint 指标日志的最后一个 `Test evaluation` 边界重新解析结果，避免把验证指标或失败尝试当成测试结果；272 组 AUC／Logloss 与 runs.csv 的 JSON 和 metrics.csv 逐项精确一致。64 个正式运行均有完成证据、协议有效标记和有限指标。旧 16 个 baseline／SISA 配对以及新 16 个 SISA seed 配对，其逐 rank 的三个划分行数、处理行数和步数一致。三轮单元均值、样本 SD 和 Logloss 差值与既有统一汇总在 1e−12 内一致。

主配对 label-macro 与 cell-macro 均正向；逐个去掉一个数据集后，其余 12 单元的 ΔAUC 仍为正：去掉 QK 为 +0.007715，去掉 KuaiRand 为 +0.005648，去掉 TAAC-25 为 +0.004236，去掉 MerRec 为 +0.005765。这是汇总敏感性检查，不是新的训练实验，也不是显著性检验。

历史测试覆盖 exact-off／zero-bias、mask 与 padding、独立初始化、可学习 λ 梯度和优化器兼容等实现性质；它们验证代码性质，不是去除某组件后的性能消融。`experiments/ablations/README.md` 当前明确没有正式完成的消融 study；配置中存在 ablation 名称、benchmark 中存在上游对照，也不能当作本项目已完成的 SISA 消融。

当前 checkout 为 `c821f798c3013beee4ffb2fddea6670de5f42047`，含工作开始前已存在的未提交修改。`unirank/pytorch/layers/attentions/sisa.py` 第 20 行的 `channel_dim: int,+2` 触发语法错误；全量 CPU 检查执行 70 项，出现 3 个模块导入错误，不能写成“当前全部回归通过”。新增证据审计测试单独运行通过；这些测试不导入模型训练模块。原有用户修改保留。历史训练成功与当前工作区能否直接启动是两个需要分别说明的事实。

验证等级为：**已归档数值和本报告汇总可核对；论文整体结论需附限制；当前 CPU 回归已通过，GPU 训练未重新执行。** 本次未启动任何 GPU 训练、重跑评测或提交集群任务。""")
    prose("claims", """## 12. 论文表述的许可范围

单元和标签都存在相关性：同一数据集的动作共享样本，同一模型的不同动作共享训练，且 baseline 只运行一轮。68 个标签组合不是 68 次独立随机重复，16 个单元也不适合未经说明地当成独立同分布样本。当前不计算 p 值、不标显著性星号、不从三点混合协议 SD 构造置信区间。
""")
    table("claims_table", "表 15：结论与证据等级", ["拟写结论", "当前判断", "建议表述或缺失证据"], [
        ["SISA 整体改善离线预测", "有单 seed 配对支持", "15/16 单元 AUC 提升；报告绝对增益及负例"],
        ["跨三 seed 显著提升", "不支持", "缺少 20262028/29 匹配 baseline；协议也未统一"],
        ["适用于已测多种架构", "有限支持", "明确 4 模型和具体接入位置，不外推所有推荐模型"],
        ["所有动作都受益", "不支持", "主配对仅 57/68 AUC 正向；保留 Offer/Purchase 等负例"],
        ["衰减或旋转是增益原因", "待受控消融", "需要组件消融和匹配参数容量控制"],
        ["额外参数较少", "日志支持", "报告总参数与稠密参数双分母"],
        ["低时延／更快／省显存", "尚无成对证据", "同硬件、相同采样、编译 warm-up 后重复测量"],
        ["线上业务指标改善", "未测", "需要在线实验或相应业务决策证据"],
    ])
    prose("next", """## 13. 投稿前应补齐的实验与材料

**P0：补齐同协议配对种子。** 如沿用已完成的 2×H100、每卡 16384 协议，先为 seeds 20262028 和 20262029 的 16 单元各补一轮 baseline，共 **32 个训练任务**，得到两轮真正配对。如果要求同协议三 seed，再为 20262027 在相同两卡协议下分别重跑 baseline 与 SISA，共再加 **32 个任务**。合计 64 个新增任务可以形成完整三轮两卡矩阵。不能把已有四卡 20262027 直接混入同协议三 seed 主统计。若改评测尾批策略，则必须对两组统一实施并重建结果版本。

**P0：封存复现版本和数据口径。** 修复当前语法错误并通过完整 CPU 回归；记录提交哈希、uv.lock、有效合并配置、数据版本与文件清单、全局及内部随机流、所有正式 job/task。更新或补充 MerRec 实际块布局说明。逐项核验论文转录来源；补充各标签正例数、正例率、历史长度分布、过滤人群比例及标签归因窗口。这些资料不能以词表大小或 loss 推测代替。

**P1：做能够区分机制的受控消融。** 所有计划仍限定在 accepted 4×4。先在 HiFormer–QK（负例）、HyFormer–TAAC-25（明显收益）和 RankMixer–KuaiRand（pooling 接入）做预声明的探索筛查，再将拟作主文结论的变体补齐适当范围与 seeds。候选包括 full SISA、去衰减、去旋转、λ 固定与可学习、generic additive-bias 参数容量控制、接入位置消融。去掉组件可能同时改变参数量，须说明如何控制；本体机制结论不能只靠 score_scale=0 的数值等价测试。

**P1：超参数与异质性。** 可测试 score_dim={8,16,32}、lambda_init={0.01,0.1,1.0}，按验证集选择而非测试集挑优；保留失败和退化配置。研究不同真实历史长度、活跃度和正例率切片；若探索 L={50,100,200}，各长度都需匹配 baseline、相同预处理与预算，并说明是否存在足够长历史。

**P1：效率。** 在相同 GPU、batch、精度和编译 warm-up 下成对测量参数数、优化器状态、峰值显存、训练 examples/s、完整 epoch 时间以及推理 p50/p95。30 秒 telemetry 只作运行记录，峰值需更密或设备内同步测量。

**P2：统计与误差分析。** 配对 seeds 完整后，报告逐 seed 的 paired Δ、均值和不确定性；如保存用户级预测，可对用户／会话做簇 bootstrap，避免把相关曝光独立重采样。只有三轮时对显著性应谨慎，预声明主指标和多重比较处理。显著性、组件归因和完整主表仍应以实际新增证据为准，本节是计划，不是已完成结果。

仍待回答的问题包括：HiFormer–QK 为何下降；MerRec 的 Offer/Purchase 为什么与 Like/Cart 不同；移除尾批评测差异后跨 seed 结论是否保持；SISA 增益是否超出单纯增加分数投影容量。""")
    prose("manuscript", """## 14. 可用于论文初稿的实验叙述

### 中文：实验设置

我们在 HiFormer、HyFormer、RankMixer 和 Zenith 四个 backbone 上评估 SISA，并采用 QK-Video、KuaiRand、TAAC-25 和 MerRec 四个多行为数据集。所有实验沿用既定预处理划分和任务标签，历史序列长度设为 100，训练一个 epoch，配置 global batch 为 32,768，采用 BF16 和分布式分块数据加载。我们在各模型已有注意力位置加入 SISA 分数偏置，主实验对 baseline 和 SISA 使用相同的实验 seed 20262027 及匹配的训练、验证和测试协议。稠密参数采用 AdamW，稀疏参数采用 Adagrad。评测指标为各动作的 ROC-AUC 与二元 Logloss；整体结果先在每个模型–数据集单元内对标签等权平均，再对 16 个单元等权平均。

### 中文：主结果和限制

在单 seed 配对实验中，SISA 在 16 个模型–数据集单元中的 15 个单元提高 AUC，单元宏平均绝对增益为 0.005841；Logloss 在全部 16 个单元下降，平均差值为 −0.002383。逐标签比较中，AUC 和 Logloss 分别在 57/68 和 55/68 个标签组合上改善。收益在 TAAC-25、KuaiRand 和 MerRec 上较大，而 QK-Video 的平均改善较小；HiFormer–QK-Video 出现负向 AUC 变化。结果说明 SISA 在已测试范围内通常能改善离线预测，但其收益与模型、数据集及动作任务相关。由于主配对仅包含一个实验 seed，这些结果为点估计，我们不据此主张统计显著性。额外两轮 SISA 采用了不同的 DDP world size 和每卡 batch，其三轮汇总仅作为描述性补充。

### English: main result with an explicit evidence boundary

In a matched single-seed evaluation across four backbones and four datasets, SISA improves cell-level macro AUC in 15 of 16 model–dataset cells, with an absolute macro gain of 0.005841. Macro Logloss decreases in all 16 cells, with an average change of −0.002383. At the model–dataset–label level, AUC and Logloss improve in 57 of 68 and 55 of 68 comparisons, respectively. Gains vary across datasets and tasks, with a negative AUC result for HiFormer on QK-Video. These findings are single-seed point estimates and do not establish statistical significance. Two additional SISA runs use a different distributed batch layout; their aggregate results are reported descriptively rather than as three matched baseline–SISA replications.

上面文字可作为初稿材料；投稿前应加入正式数据与模型引文，并用最终统一协议补跑结果更新多 seed 叙述。不要将本节视为已完成所有投稿要求的声明。""")
    prose("reproduce", """## 15. 复算与证据导航

从仓库根目录执行：

```bash
.venv/bin/python experiments/sisa_paper_report/archive/scripts/audit_sisa_paper_evidence.py
.venv/bin/python experiments/sisa_paper_report/archive/scripts/build_sisa_paper_report.py
.venv/bin/python -m unittest discover -s experiments/sisa_paper_report/archive/tests -p test_audit_sisa_paper_evidence.py -v
```

环境只通过 `uv sync --locked` 重建。审计读取已存在的正式日志和数据 footer；若这些 Git 忽略的运行时文件未迁入新机器，需要先按运行表恢复。脚本不下载数据、不启动训练、不变更原始实验 CSV。

- [单元级结果](results/cell_results.csv)：baseline、三轮 SISA、AUC／Logloss、主配对差值、三轮均值及 SD。
- [完整标签级结果](results/label_results.csv)：68 个组合的全部原始比较值。
- [正式运行映射](results/run_evidence.csv)：64 次运行、硬件、job/task、原始日志相对路径及参数量。
- [覆盖率审计](results/coverage_audit.csv)：每次运行的 train／valid／test 处理量和各 rank 步数。
- [实际合并配置](results/logged_configurations.json)与[配对配置差异](results/paired_config_differences.csv)。
- [数据元信息与 footer 行数](results/dataset_metadata.json)、[资源记录](results/resource_usage.csv)、[来源 SHA-256 清单](results/source_manifest.csv)。
- [验证说明](validation.md)记录本次可核验范围、未完成检查及报告包装状态。

本报告是冻结快照。旧 strict／expansion 文档用于追踪运行历史；本报告和上述筛选后的表格承担当前 accepted 范围的论文证据入口。""")
    prose("appendix_auc", """## 附录 A：68 个标签组合的 AUC 完整主配对

附录保留所有正向和负向结果，按数据集、动作、模型逐项查看。三轮逐 seed 数值及描述性 SD 均在 `results/label_results.csv`，不作选择性删除。""")
    for d in LABELS:
        group=[r for r in labels if r['dataset']==d]
        table("label_auc_"+d.lower(), f"附表 A：{NAMES[d]} AUC", ["模型", "标签", "Baseline", "SISA 20262027", "SISA 20262028", "SISA 20262029", "主配对 Δ", "三轮均值", "三轮 SD"], [[r['model'],r['label'],f(r['baseline_auc']),*(f(r[f'sisa_{seed}_auc']) for seed in SEEDS),sign(r['paired_delta_auc']),f(r['three_seed_mean_auc']),f(r['three_seed_sd_auc'])] for r in group])
    prose("appendix_loss", "## 附录 B：68 个标签组合的 Logloss 完整主配对\n\nLogloss 越低越好；正 Δ 表示退化。")
    for d in LABELS:
        group=[r for r in labels if r['dataset']==d]
        table("label_loss_"+d.lower(), f"附表 B：{NAMES[d]} Logloss", ["模型", "标签", "Baseline", "SISA 20262027", "SISA 20262028", "SISA 20262029", "主配对 Δ", "三轮均值", "三轮 SD"], [[r['model'],r['label'],f(r['baseline_logloss']),*(f(r[f'sisa_{seed}_logloss']) for seed in SEEDS),sign(r['paired_delta_logloss']),f(r['three_seed_mean_logloss']),f(r['three_seed_sd_logloss'])] for r in group])

    report_text = "\n\n".join(md) + "\n"
    standalone_sources = """## 15. 数据来源、完整性与使用说明

本报告以最终选定的训练任务记录、原始训练指标日志和集群运行日志为主要来源，以数据集元信息、Parquet 文件 footer 和模型实现配置为补充来源。所有 64 次正式运行的最终测试指标均已逐项核对；主文完整给出 16 个单元的结果，附录完整给出 68 个标签组合的 baseline、三轮 SISA、主配对差值、三轮均值和样本标准差。

报告中的数据规模指预处理后的交互样本；主结果指 seed 20262027 的同协议配对；补充三轮结果是混合协议下的描述性统计。本文没有使用范围外模型或数据集支持结论，没有把测试指标用于配置挑选，也没有把未完成的消融、效率测量或在线实验写成既有结果。

本报告可独立阅读和复制。论文写作时可以直接使用“实验设置汇总”和第 14 节初稿段落，同时保留单 seed、尾批覆盖变化、未进行统计显著性检验等限制。正式发表前仍需核验原论文引文、补齐同协议配对种子，并完成前述待补实验。

"""
    report_text = re.sub(r"## 15\. 复算与证据导航.*?(?=## 附录 A)", standalone_sources, report_text, flags=re.S)
    report_text = report_text.replace("精确任务映射和原始日志路径见 `results/run_evidence.csv`。", "这些任务标识用于区分最终正式结果与重复或失败尝试，不增加实验种子数。")
    report_text = report_text.replace("`experiments/ablations/README.md` 当前明确没有正式完成的消融 study", "当前实验档案中没有正式完成的 SISA 性能消融研究")
    report_text = report_text.replace("三轮逐 seed 数值及描述性 SD 均在 `results/label_results.csv`，不作选择性删除。", "下表同时列出三轮逐 seed 数值及描述性 SD，不作选择性删除。")
    report_text = re.sub(r"当前 checkout 为 .*?原有用户修改保留。", "首次检查时，工作区的 SISA 源码存在语法错误，造成 3 个模块导入错误。后续整理已修复该错误，完整 CPU 回归检查 88 项通过，独立结果审计测试 5 项通过。CPU 回归和历史指标复核仍不等同于重新完成 GPU 训练复现。", report_text, flags=re.S)
    report_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", report_text)
    if re.search(r"https?://|\[[^\]]+\]\(|详见|见同目录|结果见.*文件", report_text):
        raise ValueError("Standalone report must contain no links or external reading dependencies")
    (STUDY/'report.md').write_text(report_text)
    print(f"Built standalone report: {STUDY.relative_to(ROOT)}/report.md; no hyperlinks")
    return



if __name__ == '__main__':
    main()
