"""Build the RISE failure analysis from the frozen experiment evidence."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TITLE = "RISE 失败案例分析：HiFormer 在 QK-Video 上为何下降"
STAMP = datetime.now(timezone.utc).isoformat()
FILES = {
    "cells": "experiments/sisa_three_seed_unified/results/integrated_cells.csv",
    "labels": "experiments/sisa_three_seed_unified/results/label_comparison.csv",
    "placements": "experiments/rise_sample_analysis/results/targeted_training_comparisons.csv",
    "placement_runs": "experiments/rise_sample_analysis/results/targeted_training_runs.csv",
    "placement_evidence": "experiments/rise_sample_analysis/results/targeted_training_evidence.json",
    "groups": "experiments/rise_sample_analysis/results/hiformer_QK_Video_Action_groups_v1/group_results.csv",
    "group_manifest": "experiments/rise_sample_analysis/results/hiformer_QK_Video_Action_groups_v1/manifest.json",
    "replication": "experiments/rise_sample_analysis/results/site_replication_comparisons.csv",
    "replication_evidence": "experiments/rise_sample_analysis/results/site_replication_evidence.json",
    "model": "model_zoo/HiFormer.py",
    "placement_code": "experiments/rise_sample_analysis/scripts/targeted_controls.py",
}


def csv_rows(key):
    with (ROOT / FILES[key]).open() as handle:
        return list(csv.DictReader(handle))


def number(value, signed=False, digits=8):
    return format(float(value), f"{'+' if signed else ''}.{digits}f")


cells = csv_rows("cells")
assert len(cells) == 20
assert sum(float(row["auc_delta_vs_baseline"]) > 0 for row in cells) == 19
negative = [row for row in cells if float(row["auc_delta_vs_baseline"]) < 0]
assert [(r["model"], r["dataset_id"]) for r in negative] == [("HiFormer", "QK_Video_Action")]
cell = negative[0]
labels = [r for r in csv_rows("labels") if r["model"] == "HiFormer" and r["dataset"] == "QK_Video_Action"]
placements = [r for r in csv_rows("placements") if r["model"] == "HiFormer"]
groups = csv_rows("groups")
group_manifest = json.loads((ROOT / FILES["group_manifest"]).read_text())
replication = [r for r in csv_rows("replication") if r["label"] == "__macro__"]
target = next(r for r in placements if r["mode"] == "target_only" and r["label"] == "__macro__")
field = next(r for r in placements if r["mode"] == "field_only" and r["label"] == "__macro__")
assert all(float(r["delta_AUC_vs_Baseline"]) > 0 for r in placements if r["mode"] == "target_only")
assert all(float(r["delta_AUC_vs_Baseline"]) < 0 for r in placements if r["mode"] == "field_only")

sources = []
for key, path in FILES.items():
    source = {"id": key, "label": {
        "cells": "正式二十组结果的冻结汇总", "labels": "HiFormer QK-Video 标签级冻结结果",
        "placements": "HiFormer QK-Video 接入位置控制", "placement_runs": "位置控制作业与参数量审计",
        "placement_evidence": "位置控制日志及协议审计", "groups": "QK-Video 同一样本配对分组结果",
        "group_manifest": "QK-Video 预测复现与样本分组清单", "replication": "Zenith 与 HyFormer 位置控制",
        "replication_evidence": "跨模型位置控制审计", "model": "HiFormer 前向计算实现",
        "placement_code": "位置控制开关和初始化核查实现",
    }[key], "path": path}
    if path.endswith(".csv"):
        source["query"] = {
            "engine": "duckdb", "language": "sql",
            "sql": f"SELECT * FROM read_csv_auto('{path}', header = true)",
            "description": source["label"], "tables_used": [path],
            "filters": ["正式范围仅含五个模型和四个数据集；失败案例为 HiFormer 与 QK-Video"],
            "metric_definitions": [
                "宏 AUC 和宏 LogLoss 为本单元标签的等权算术平均；AUC 越高越好，LogLoss 越低越好",
                "所有差值为 RISE 或指定变体减去表中明确列出的 Baseline；单位为绝对原始尺度",
            ],
        }
    sources.append(source)

blocks = [{"id": "title", "type": "markdown", "body": f"# {TITLE}"}]
tables = []
datasets = {}
markdown_parts = [f"# {TITLE}\n\n核查日期：2026 年 9 月 24 日。"]


def prose(identifier, title, body, source=None):
    text = f"## {title}\n\n{body}"
    block = {"id": identifier, "type": "markdown", "body": text}
    if source:
        block["sourceId"] = source
    blocks.append(block)
    markdown_parts.append(text)


def table(identifier, title, columns, rows, source, subtitle=""):
    datasets[identifier] = rows
    cols = [{"field": key, "label": label, "type": "text", **({"align": "right"} if key != columns[0][0] else {})} for key, label in columns]
    tables.append({
        "id": identifier, "title": title, "subtitle": subtitle, "showDescription": bool(subtitle),
        "dataset": identifier, "sourceId": source, "layout": "full", "density": "comfortable",
        "defaultSort": {"field": columns[0][0], "direction": "asc"}, "columns": cols,
    })
    blocks.append({"id": f"{identifier}_block", "type": "table", "tableId": identifier})
    markdown_parts.append("\n".join([
        f"**{title}**", "", "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *["| " + " | ".join(str(row[key]) for key, _ in columns) + " |" for row in rows],
        *(["", subtitle] if subtitle else []),
    ]))


prose("summary", "核心判断：问题主要定位于字段交互接入",
    "**唯一下降组合是 HiFormer × QK-Video。现有接入位置控制支持：完整 RISE 的负向结果主要与字段交互层接入有关，历史汇聚位置本身可以产生正向收益。**\n\n"
    "在与追加对照对应的同一训练种子和预算下，仅保留历史汇聚接入后，AUC 超过原生 Baseline；仅保留字段交互接入时，AUC 进一步下降。因此，失败案例应解释为 RISE 与该模型特定接入位置及任务的适配问题。\n\n"
    "代码提供了一个具体机制假说：字段层处理的是用户特征、候选特征、历史汇聚和任务 token，而 RISE 沿其固定排列累积衰减与相位；这套顺序结构可能干扰字段与任务之间的交互。**位置证据已经存在，但字段顺序、衰减、旋转、参数容量和优化适应的独立作用尚未被分离。**")

prose("definitions", "先区分整体汇总与机制对照的 Baseline",
    "正式范围为 HiFormer、HyFormer、OneTrans、RankMixer、Zenith 在 MerRec、QK-Video、KuaiRand、TencentGR 上的 20 个模型—数据集单元。当前冻结汇总中，19 组宏 AUC 高于各自 Baseline，唯一负向单元为 HiFormer–QK-Video。\n\n"
    "QK-Video 的宏 AUC 与宏 LogLoss 分别是 click、follow、like、share 四个标签的等权平均；AUC 越高越好，LogLoss 越低越好。下文 Δ 均为 RISE 或指定变体减 Baseline，使用绝对 AUC 单位。\n\n"
    "整体汇总比较历史单次 Baseline 与 RISE 三次运行均值；位置控制则比较 seed 20262028 对应的追加原生 Baseline 与完整模型、两种位置变体。两套 Baseline 不同，不能交叉代入。三次 RISE 运行的训练协议并非严格同构，也不能当作三次严格配对重复。", "cells")
comparison_rows = [
    {"basis": "1. 当前整体汇总：三次 RISE 均值", "baseline": number(cell["baseline_auc"]), "rise": number(cell["latest_three_seed_mean_auc"]), "delta": number(cell["auc_delta_vs_baseline"], True), "baseline_ll": number(cell["baseline_logloss"]), "rise_ll": number(cell["latest_three_seed_mean_logloss"])},
    {"basis": "2. 早期主配对：seed 20262027", "baseline": number(cell["baseline_auc"]), "rise": number(cell["latest_20262027_auc"]), "delta": number(float(cell["latest_20262027_auc"])-float(cell["baseline_auc"]), True), "baseline_ll": number(cell["baseline_logloss"]), "rise_ll": number(cell["latest_20262027_logloss"])},
]
table("headline_comparison", "整体结果与早期配对结果", [("basis", "比较口径"), ("baseline", "Baseline AUC"), ("rise", "RISE AUC"), ("delta", "ΔAUC"), ("baseline_ll", "Baseline LogLoss"), ("rise_ll", "RISE LogLoss")], comparison_rows, "cells")
prose("headline_interpretation", "下降方向重复出现，但不能据此宣称统计显著",
    "三次完整 RISE 的宏 AUC 分别为 0.93106300、0.93089125、0.93061050，均低于整体汇总中的历史 Baseline。最新三次均值的 AUC 下降 0.00230833，约为 0.230833 个 AUC 百分点；宏 LogLoss 同时上升 0.00023858。\n\n"
    "早期 seed 20262027 主配对的宏 LogLoss 曾有极小改善，但该结论不能套用到当前三次均值。运行间协议差异、缺少三个对应 Baseline 和测试集已参与探索，均限制了统计推断。", "cells")

follow = next(r for r in labels if r["label"] == "follow")
share = (float(follow["sisa_mean_delta_auc_vs_baseline"]) / 4) / float(cell["auc_delta_vs_baseline"])
prose("label_story", "follow 贡献约三分之二的宏 AUC 损失",
    f"三次均值中，四个标签的 AUC 均低于历史 Baseline；follow 下降最大，为 {number(follow['sisa_mean_delta_auc_vs_baseline'], True)}。它对宏 AUC 的贡献为该差值除以四，占净下降的 **{share:.2%}**。这定位了失败最集中的任务，但不等于已经解释其成因。", "labels")
label_rows = [{"label": r["label"], "baseline": number(r["baseline_auc"]), "rise": number(r["sisa_mean_auc"]), "delta": number(r["sisa_mean_delta_auc_vs_baseline"], True), "macro": number(float(r["sisa_mean_delta_auc_vs_baseline"])/4, True)} for r in labels]
table("label_table", "当前三次均值的逐标签 AUC", [("label", "标签"), ("baseline", "Baseline"), ("rise", "RISE 三次均值"), ("delta", "ΔAUC"), ("macro", "对宏 ΔAUC 的贡献")], label_rows, "labels")

prose("placement_story", "接入位置对照把负面来源定位到字段层",
    "**只在历史汇聚接入 RISE 时超过 Baseline；只在字段交互接入时下降更大。** 下表使用 seed 20262028 的对应 Baseline，四种配置保持训练预算与记录中的样本覆盖对齐。完整 RISE 同时包含一个历史汇聚位置和三个字段交互位置。", "placements")
placement_rows = [
    {"metric": "1. 宏 AUC", "baseline": number(target["Baseline_AUC"]), "full": number(target["Full_RISE_AUC"]), "target": number(target["Targeted_AUC"]), "field": number(field["Targeted_AUC"])},
    {"metric": "2. ΔAUC vs Baseline", "baseline": number(0, True), "full": number(float(target["Full_RISE_AUC"])-float(target["Baseline_AUC"]), True), "target": number(target["delta_AUC_vs_Baseline"], True), "field": number(field["delta_AUC_vs_Baseline"], True)},
    {"metric": "3. 宏 LogLoss", "baseline": number(target["Baseline_logloss"]), "full": number(target["Full_RISE_logloss"]), "target": number(target["Targeted_logloss"]), "field": number(field["Targeted_logloss"])},
]
table("placement_table", "HiFormer–QK-Video 的位置控制", [("metric", "指标"), ("baseline", "Baseline"), ("full", "完整 RISE"), ("target", "仅历史汇聚"), ("field", "仅字段交互")], placement_rows, "placements", "三种 RISE 配置分别重新训练；这里只比较同一 seed 的探索性控制。")
prose("placement_reading", "字段接入的负向变化在两组参照中同向",
    "从原生 Baseline 到“仅字段交互”，宏 AUC 下降 0.00404725；从“仅历史汇聚”到完整 RISE，宏 AUC 下降 0.00352350。反过来，历史汇聚接入相对 Baseline 提高 0.00097675；完整 RISE 相对“仅字段交互”提高 0.00150050。**两个参照下的方向一致，支持字段接入是这组配置的重要负面来源。** 这些差异来自分别训练的模型，不能线性相加成固定组件贡献，也不能据此宣称严格统计交互。\n\n"
    "follow 的位置响应最明显：Baseline 为 0.921287，完整 RISE 为 0.913586；仅历史汇聚恢复到 0.921751，仅字段交互则为 0.912760。仅历史汇聚的四个标签 AUC 均超过对应 Baseline，但 follow 的 LogLoss 仍略高 0.000010，因此不能写成所有标签、所有指标都改善。", "placements")

prose("mechanism", "最合理的机制解释：字段语义与累积顺序可能不匹配",
    "HiFormer 先用目标物品对行为历史做注意力汇聚，再把用户上下文、目标物品特征、历史汇聚特征和任务 token 拼在一起，经三层交互后由任务 token 输出各标签预测。历史汇聚中的 key 对应行为序列；后面的字段 token 则对应不同特征与任务，其排列没有同样的行为时间含义。\n\n"
    "RISE 在后者仍使用沿 token 排列累计的衰减和相位，并将附加分数加到原生 attention score。这里的顺序结构可能改变字段和任务 token 的相互关注。原生 HiFormer 的 Q/K/V 还由整段 token 展平后做低秩投影得到，已经显式混合字段信息，因此叠加的累积偏置可能与原有交互方式发生冲突。\n\n"
    "位置实验与这一解释相容，但尚未把“非时间字段排列”与衰减、相位、内容投影、参数量变化、初始化及优化适应分开。最稳妥的结论是字段接入不适配；把它进一步归结为某一个数学部件，需要字段侧的专项控制。", "model")

prose("groups_story", "短历史点估计更差，稀少的 follow 是敏感任务",
    "对 seed 20262028 的 Baseline 和完整 RISE，在同一批 48,529,408 条测试交互上重新计算配对指标。follow 正例为 40,518 条，占 0.08349%；share 正例占 0.10119%，同样稀少。因此可以说 follow 是稀少且损失集中的任务，不能仅凭稀少性解释它为什么下降最多。\n\n"
    "按有效历史长度分组后，四个非空长度组的宏 AUC 差值和 follow 差值都为负，短历史组的负向点估计更大。该现象与“历史信号不足时更难抵消字段侧不利变化”的解释相容，但长度、用户组成、样本难度等因素共同变化，没有独立操纵。", "group_manifest")
length_rows = []
for index, group in enumerate(["1-20", "21-50", "51-99", "100"], 1):
    selected = [r for r in groups if r["dimension"] == "effective_length" and r["group"] == group]
    macro = next(r for r in selected if r["label"] == "macro_4_tasks")
    follow_group = next(r for r in selected if r["label"] == "follow")
    length_rows.append({"length": f"{index}. {group}", "rows": f"{int(macro['rows']):,}", "positive": f"{int(follow_group['positives']):,}", "macro": number(macro["delta_auc"], True), "follow": number(follow_group["delta_auc"], True)})
table("length_table", "有效历史长度分层", [("length", "有效历史长度"), ("rows", "测试交互数"), ("positive", "follow 正例"), ("macro", "宏 ΔAUC"), ("follow", "follow ΔAUC")], length_rows, "groups", "配对预测对应 seed 20262028；分组内 AUC 的加权平均不等于全局 AUC。")
prose("groups_limits", "分层结果用于描述条件差异，不是成因证明",
    "这些长度组是同一测试集的切分，并不提供多个训练重复；现有用户簇不确定性估计也不能替代训练种子的变异评估。组间点估计大小不构成组间差异显著性的检验。部分频次分组并非负向，不能把长度分层的同向结果泛化为“所有人群、所有切法均退化”。", "groups")

prose("counterexamples", "为何只有这一组下降：架构与数据条件共同决定净收益",
    "QK-Video 上其他四个模型的完整 RISE 宏 AUC 均提升，HiFormer 在另外三个数据集上也提升。因而“QK-Video 不适合 RISE”或“HiFormer 不适合 RISE”都超出证据。下面保留这两类反例：它们要求解释具体的架构、接入位置与数据条件，而不是把原因单独归给模型或数据集。", "cells")
counter_rows = []
for r in cells:
    if r["model"] == "HiFormer" or r["dataset_id"] == "QK_Video_Action":
        dataset = "TencentGR" if r["dataset_id"] == "TencentGR_10M_Action" else r["dataset"]
        counter_rows.append({"unit": f"{r['model']} · {dataset}", "baseline": number(r["baseline_auc"]), "rise": number(r["latest_three_seed_mean_auc"]), "delta": number(r["auc_delta_vs_baseline"], True)})
table("counter_table", "当前整体汇总中的相关对照", [("unit", "模型 · 数据集"), ("baseline", "Baseline AUC"), ("rise", "RISE 三次均值"), ("delta", "ΔAUC")], counter_rows, "cells", "均为当前冻结汇总的描述性结果；不用于严格跨协议显著性推断。")
prose("cross_placement", "跨模型位置控制支持位置敏感性，但也保留反例",
    "QK-Video 上的追加控制显示，Zenith 也出现“仅历史位置为正、仅 token 交互为负”，但完整 RISE 仍为正向；说明字段/token 位置的不利影响可以被其他位置的收益抵消。HyFormer 的序列编码与 query 解码两个单位置控制均为正向，完整双阶段略高于两者；其两个位置均处理行为序列信息，不能把所有 sequence-end 接口都解释为非时间字段。\n\n"
    "因此，对“唯一下降”的可支持解释是：在 HiFormer–QK-Video 这组已测配置中，字段接入的不利变化超过历史汇聚带来的有利变化，表现为完整模型净下降；它不是一个跨模型、跨数据集都必然成立的规律。", "replication")
rep_rows = []
for r in replication:
    rep_rows.append({"config": f"{r['model']} · {r['mode']}", "baseline": number(r["Baseline_AUC"]), "full": number(r["Full_RISE_AUC"]), "site": number(r["Placement_AUC"]), "delta": number(r["delta_AUC_vs_Baseline"], True)})
table("replication_table", "QK-Video 上的其他模型位置控制", [("config", "模型 · 单位置配置"), ("baseline", "Baseline"), ("full", "完整 RISE"), ("site", "单位置 AUC"), ("delta", "单位置 ΔAUC")], rep_rows, "replication")

prose("limitations", "结论边界：已经定位位置，尚未分离微观机制",
    "1. **整体汇总、早期主配对和 seed 20262028 位置控制属于不同口径。** 本文逐处明确参照，不能混合为严格配对的三种子检验。\n"
    "2. **位置控制是单训练种子的事后探索。** 活跃稠密参数量随位置变化；相同预算与保留位置初始化的一致性，不能消除参数容量和优化路径变化。\n"
    "3. **只有一轮训练的末端验证不足以诊断收敛和过拟合。** 不能把当前下降直接归为已证实的过拟合、训练不足或随机波动。\n"
    "4. **follow 正例稀少是已测事实，稀少导致退化仍是未验证解释。** share 也稀少，且任务目标、正负样本结构、排序难度等可能共同影响响应。\n"
    "5. **测试集已经用于诊断和选择后续方案。** 保留原始完整 RISE 负例；不能将事后挑选的仅历史汇聚结果替换回正式主表。\n"
    "6. **参数值不等于实际扰动幅度。** 仅凭 head scale 或衰减参数不能证明附加分数过强，需要在对应失败模型中直接观察分数、注意力和排序变化。")

prose("next", "下一步应确认定位并分离字段侧机制",
    "1. 优先在一致协议下补充 Baseline、完整 RISE、仅历史汇聚的成对训练重复，以验证位置结论是否稳定；保留四标签与宏平均结果，尤其监控 follow。\n"
    "2. 固定历史侧配置，在字段侧一次只改变一个因素：内容投影控制、取消累积衰减、取消相位、降低附加分数强度，或改变字段排列并保持其他条件可比。用验证集选方案，独立留出确认结果。\n"
    "3. 在同一批样本上测量字段层实际附加分数相对原生分数的尺度、注意力分配变化和 follow 排序变化，检查是否存在任务 token 或字段对的集中扰动。\n\n"
    "仍需回答的核心问题是：字段侧损失由哪一项构造驱动、为什么 follow 尤其敏感，以及单位置收益能否跨种子保留。本报告仅分析已有证据，没有启动训练或修改模型。")

prose("paper_text", "可直接用于论文的失败案例表述",
    "在五个模型与四个数据集构成的 20 个实验单元中，HiFormer 在 QK-Video 上是完整 RISE 宏 AUC 唯一下降的组合。按当前汇总口径，其宏 AUC 从 0.93316325 降至 0.93085492，损失主要集中于 follow 任务。进一步的单种子接入位置控制表明，相对于对应原生 Baseline 的 0.93343800，仅在历史汇聚阶段接入 RISE 可将宏 AUC 提高至 0.93441475，而仅在字段交互层接入则降至 0.92939075；完整 RISE 为 0.93089125。该结果支持字段交互接入是此失败案例的重要负面来源。HiFormer 的字段层混合用户特征、候选特征、历史汇聚表示与任务 token，其排列不具有行为时间序列的同等含义；沿该排列引入累积偏置可能与既有字段交互机制不匹配。由于位置控制同时改变活跃参数量，且目前只有一个训练种子的探索性结果，这一解释尚不能进一步唯一归因于字段顺序、衰减或相位。该负例表明接入位置影响效果，并提示需考虑 token 语义与骨干信息流的匹配；不能假设所有 attention 位置均受益。")

# Keep the delivery concise: eight evidence-led sections and five exact-lookup tables.
paper_paragraph = next(block["body"].split("\n\n", 1)[1] for block in blocks if block["id"] == "paper_text")
blocks = [{"id": "title", "type": "markdown", "body": f"# {TITLE}"}]
markdown_parts = [f"# {TITLE}\n\n核查日期：2026 年 9 月 24 日。"]
table_lookup = {item["id"]: item for item in tables}
tables = []


def reuse_table(identifier):
    item = table_lookup[identifier]
    columns = [(c["field"], c["label"]) for c in item["columns"]]
    table(identifier, item["title"], columns, datasets[identifier], item["sourceId"], item.get("subtitle", ""))


prose("summary", "一、负效应已经定位到字段交互接入配置",
    "**HiFormer × QK-Video 是当前 20 组结果中完整 RISE 宏 AUC 唯一下降的组合。已有位置控制表明：仅历史汇聚接入可以提高性能，字段交互接入会产生明显负向变化；两者共同接入后的净结果为下降。**\n\n"
    "这已经超出纯推测：字段接入的不利表现有实际对照支持。但其内部原因尚未唯一确定。字段/任务 token 的排列缺少行为时间语义，沿此排列施加累积偏置可能与 HiFormer 的全局字段交互冲突，是与代码和位置结果相容的解释，仍需专项控制确认。")
prose("scope", "二、整体三次均值与位置对照使用不同 Baseline",
    "正式范围为 HiFormer、HyFormer、OneTrans、RankMixer、Zenith × MerRec、QK-Video、KuaiRand、TencentGR，共 20 个单元；当前 19 组宏 AUC 为正向，只有 HiFormer–QK-Video 为负向。这里的“19 组提升”仅指宏 AUC；宏 LogLoss 改善为 17/20 组，也不表示所有标签都改善。QK 的宏指标是 click、follow、like、share 的等权平均；Δ 均为 RISE 减参照，使用绝对 AUC 单位。\n\n"
    "当前整体汇总用历史单次 Baseline 对比三次 RISE 均值；追加位置控制使用 seed 20262028 对应的另一个 Baseline，不能交叉代入。三次完整 RISE 宏 AUC 分别为 0.93106300、0.93089125、0.93061050，均低于历史 Baseline，但协议不完全相同，尚未形成三个 seed 各自成对的完整重复（seed 20262028 已有追加 Baseline），不能当作三个严格配对重复。", "cells")
reuse_table("headline_comparison")
prose("labels", "三、follow 贡献 68.16% 的净宏 AUC 损失",
    "当前三次均值中四标签 AUC 都下降，follow 的下降最突出，占净宏损失的 68.16%；宏贡献按标签差值除以四计算。它定位了重点受损任务，不直接说明成因。当前三次均值的宏 LogLoss 也升高 0.00023858；早期单次配对的极小 LogLoss 改善不能沿用到当前均值。", "labels")
reuse_table("label_table")
prose("placements", "四、关闭字段侧后，AUC 从下降转为提升",
    "位置对照使用 seed 20262028、2 卡、每卡 batch 16384、global batch 32768、1 epoch、最大历史长度 100；记录中的训练、验证、测试覆盖分别为 394,428,416、50,036,736、48,529,408 条。完整 RISE 接入一个历史汇聚位置和三个字段交互层。四种配置按已记录的协议和覆盖对齐，不表示历史代码与数据字节已完全追溯验证。\n\n"
    "只在字段侧接入相对 Baseline 下降 0.00404725；在仅历史汇聚的基础上加入字段侧，下降 0.00352350。反向看，历史汇聚接入在两个参照下都带来正向变化。因此，**字段交互接入是本配置重要的负面来源**。这些是分别训练后的配置差异，不能线性相加成固定组件贡献。", "placements")
reuse_table("placement_table")
prose("placement_follow", "follow 同样随字段接入而恶化、随仅历史接入而恢复",
    "seed 20262028 下，follow AUC 为 Baseline 0.921287、完整 RISE 0.913586、仅历史汇聚 0.921751、仅字段交互 0.912760。仅历史汇聚的四标签 AUC 都高于对应 Baseline，但 follow LogLoss 仍略高 0.000010；不能写成全部任务、全部指标改善。", "placements")
# Keep the label-level placement interpretation subordinate to section four.
blocks[-1]["body"] = blocks[-1]["body"].replace("## follow", "### follow", 1)
markdown_parts[-1] = blocks[-1]["body"]
prose("mechanism", "五、内部解释指向 token 语义与交互方式的不匹配",
    "HiFormer 先对行为历史做目标注意力汇聚，再将用户上下文、候选特征、历史汇聚和任务 token 拼接。后面的三层处理字段与任务关系，排列并不具有行为时间的同等含义。原生 Q/K/V 将全段 token 展平后做全局低秩投影，RISE 则从逐 token 表示构造修正并沿排列累计衰减与相位；两种构造可能对字段关系施加不相容的偏好。\n\n"
    "位置控制支持“字段接入配置不适配”，但未把字段顺序、衰减、相位、内容投影、容量和优化适应分开。不能直接写成“已证明旋转有害”或“已证明时间偏置造成下降”。", "model")
prose("groups", "六、短历史更敏感，稀少性不足以单独解释 follow",
    "已有配对预测分析覆盖 48,529,408 条测试交互，本次核对其冻结结果：四个非空历史长度组的宏 AUC 与 follow AUC 差值都为负，短历史组的负向点估计更大。follow 有 40,518 个正例，占 0.08349%；share 同样稀少，占 0.10119%。因此，follow 的稀少性是事实，却不足以单独解释其更大损失。\n\n"
    "以下是固定模型的描述性分层：组间点估计不构成组间显著性检验，组内 AUC 加权平均不等于全局 AUC；部分频次组并非下降，不能泛化为所有人群都受损。", "group_manifest")
reuse_table("length_table")
prose("uniqueness", "七、主表唯一负向，不代表其他模型没有局部负效应",
    "QK-Video 上其余四个模型、HiFormer 在其余三个数据集上的完整 RISE 宏 AUC 都为正向。更直接的反例是 Zenith：仅 token 交互接入也为负向，但完整 RISE 仍为正向，说明其他位置的收益可以抵消局部损失。HyFormer 的两个位置都处理行为序列，其单位置控制均为正向，不能把所有 sequence-end 参考方式等同于非时间字段。\n\n"
    "因此，HiFormer–QK-Video 的唯一净下降反映了该骨干、接入位置与数据条件共同作用后的最终表现；不能归结为 QK-Video 或 HiFormer 普遍不适用。下面的跨模型控制同为单种子探索，不能据此推出普遍规律。")
reuse_table("replication_table")
prose("boundaries", "八、可用于论文的结论与尚未解决的问题",
    "位置控制为单种子的事后探索，且活跃稠密参数量不同：Baseline 27,824,132、完整 RISE 27,889,090、仅历史汇聚 27,826,102、仅字段交互 27,887,120。保留位置初始化与已记录协议的对齐，不消除容量及优化路径变化。只有一轮训练末端评测不足以证明过拟合或未收敛。主表应保留完整 RISE 负例，不能用事后选择的仅历史结果替换。\n\n"
    "优先补充一致协议的成对训练重复确认位置结论，再在字段侧分别控制内容投影、衰减、相位与排列，并观察实际附加分数和 follow 排序变化；这些工作尚未由本报告执行。\n\n"
    "### 可直接复制的论文表述\n\n" + paper_paragraph)
datasets = {item["dataset"]: datasets[item["dataset"]] for item in tables}
datasets["placement_deltas"] = [
    {"configuration": "Baseline", "delta_auc": 0.0, "auc": float(target["Baseline_AUC"]), "logloss": float(target["Baseline_logloss"])},
    {"configuration": "完整 RISE", "delta_auc": float(target["Full_RISE_AUC"])-float(target["Baseline_AUC"]), "auc": float(target["Full_RISE_AUC"]), "logloss": float(target["Full_RISE_logloss"])},
    {"configuration": "仅历史汇聚", "delta_auc": float(target["delta_AUC_vs_Baseline"]), "auc": float(target["Targeted_AUC"]), "logloss": float(target["Targeted_logloss"])},
    {"configuration": "仅字段交互", "delta_auc": float(field["delta_AUC_vs_Baseline"]), "auc": float(field["Targeted_AUC"]), "logloss": float(field["Targeted_logloss"])},
]
placement_chart = {
    "id": "placement_auc_delta", "title": "接入位置与宏 AUC 差值",
    "subtitle": "seed 20262028；相对对应 Baseline 的绝对 AUC 差值，零线为 Baseline",
    "type": "horizontalBar", "intent": "comparison", "question": "不同接入位置相对同一 Baseline 的变化方向和幅度如何？",
    "rationale": "四种配置的单一同尺度指标，带零点和明确正负符号；对应表格保留准确查值。",
    "dataset": "placement_deltas", "sourceId": "placements", "palette": {"kind": "diverging", "midpoint": 0},
    "labels": {"values": "auto"}, "layout": "full", "maxRows": 4, "valueFormat": "number",
    "encodings": {
        "x": {"field": "configuration", "type": "nominal", "aggregate": "none", "label": "接入位置"},
        "y": {"field": "delta_auc", "type": "quantitative", "aggregate": "none", "label": "ΔAUC（绝对值）"},
        "tooltip": [{"field": "auc", "type": "quantitative", "label": "宏 AUC"}, {"field": "logloss", "type": "quantitative", "label": "宏 LogLoss"}],
    }, "xAxisTitle": "接入位置", "yAxisTitle": "ΔAUC（绝对值）",
}
placement_position = next(index for index, block in enumerate(blocks) if block["id"] == "placement_table_block")
blocks.insert(placement_position, {"id": "placement_chart_block", "type": "chart", "chartId": "placement_auc_delta"})

artifact = {
    "surface": "report",
    "manifest": {"version": 1, "surface": "report", "title": TITLE,
        "description": "以二十组冻结结果、标签分解、接入位置控制和配对样本分层解释唯一负向单元。",
        "generatedAt": STAMP, "blocks": blocks, "cards": [], "charts": [placement_chart], "tables": tables, "sources": sources},
    "snapshot": {"version": 1, "generatedAt": STAMP, "status": "ready", "datasets": datasets},
    "sources": sources,
}
(OUT / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
(OUT / "report.md").write_text("\n\n".join(markdown_parts) + "\n")
evidence = {"generated_at": STAMP, "scope_cells": len(cells), "positive_cells": 19,
    "negative_cell": {"model": "HiFormer", "dataset": "QK-Video"}, "follow_net_loss_share": share,
    "sources": {k: {"path": v, "sha256": hashlib.sha256((ROOT/v).read_bytes()).hexdigest()} for k, v in FILES.items()},
    "supporting_datasets": datasets}
(OUT / "evidence_summary.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
(OUT / "source_notes.md").write_text("""# Report contract and QA notes

Audience: technical. Delivery: portable HTML. Markdown is the same narrative and table source for copying, not a second runtime.

Question: Why is HiFormer on QK-Video the only negative AUC cell in the accepted 20-cell RISE matrix?
Answer: position controls locate a major negative source in field interaction; detailed order/decay/phase causes remain unresolved.

Required structure mapping: title and technical summary; scope/metric definitions moved ahead of evidence; results and experimental design are interleaved in the placement section; uncertainty and limitations have a dedicated section; recommendations and remaining questions are merged; a paper-ready paragraph repeats the validated conclusion for direct reuse.

Visual contract: exact lookup and distinctions among comparison baselines are the purpose. Five full-width source-backed native tables have adjacent interpretation, signed absolute AUC differences, neutral numeric colors and explicit default sorting. The canonical report validator rejects table-only reports, so a single native horizontal bar comparison shows the four placement configurations against the same zero reference. It uses a diverging palette and explicit configuration labels, retains baseline AUC and LogLoss for inspection, and is followed by the exact table. Raw unrounded values are preserved in original CSV sources; displayed tables use eight decimal places for AUC/LogLoss.

No significance assertion is made from three heterogeneous RISE runs or one-seed position controls. Group intervals, if inspected in source CSV, concern fixed-model paired test predictions and are not evidence for training-seed stability. No claim of universal subgroup decline, proven overfitting, causal sparsity, or isolated decay/phase causality is made.

Additional current-turn audits: artifacts/hiformer_qk_failure_20260924/group_audit.json and artifacts/hiformer_qk_failure_20260924/site_audit.json.

Sources retain their historical filenames. All reader-facing method names are RISE. Existing reports, models and training jobs are unchanged.

Reproduce from repository root with the repository virtual environment:
`.venv/bin/python experiments/hiformer_qk_diagnostic/failure_analysis_20260924/build_report.py`

Package through the Data Analytics plugin's canonical `npm run report:deliver -- --input <artifact.json> --output <report.html>` command. Keep its final receipt as `delivery_receipt.json`. Do not manually edit generated HTML.
""")
print(json.dumps({"tables": len(tables), "blocks": len(blocks), "datasets": {k:len(v) for k,v in datasets.items()}, "follow_loss_share": share}, ensure_ascii=False))
