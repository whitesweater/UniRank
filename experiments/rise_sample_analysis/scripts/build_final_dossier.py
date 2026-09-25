"""Build a self-contained academic dossier from completed, audited evidence.

The initial feasibility report is preserved. This writes a separate final report
and paper insertion fragment; it never changes training or frozen source data.
"""

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY / "results"
SOURCE_PATHS = set()


def read_rows(relative):
    path = RESULTS / relative
    SOURCE_PATHS.add(path)
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(relative):
    path = RESULTS / relative
    SOURCE_PATHS.add(path)
    return json.loads(path.read_text())


def number(value, digits=6, signed=False):
    if value is None or value == "":
        return "不可定义"
    return f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"


def interval(row):
    if row.get("auc_ci_lower") in (None, ""):
        return "不可定义"
    return f"[{number(row['auc_ci_lower'])}, {number(row['auc_ci_upper'])}]"


def table(headers, records):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(str(value) for value in row) + " |" for row in records]) + "\n"


def embedded_note(name):
    path = STUDY / "notes" / name
    SOURCE_PATHS.add(path)
    lines = path.read_text().splitlines()
    return "\n".join(("##" + line if line.startswith("#") else line) for line in lines[1:])


def group_name(name):
    translations = {"train_target_unseen": "训练目标未见", "0": "0", "100": "100"}
    return translations.get(name, name.replace("positive_le_", "1–").replace("gt_", ">").replace("_le_", "至"))


def main():
    baseline = read_rows("new_training_runs.csv")
    targeted = read_rows("targeted_training_runs.csv")
    replication = read_rows("site_replication_runs.csv")
    site_comparisons = read_rows("site_replication_comparisons.csv")
    site_labels = read_rows("site_replication_label_metrics.csv")
    if len(replication) != 6 or not all(r["eligible"] == "True" for r in replication):
        raise ValueError("All six cross-model placement runs must pass their audit")
    if len(baseline) != 5 or len(targeted) != 3 or not all(r["eligible"] == "True" for r in baseline + targeted):
        raise ValueError("All eight planned new training runs must pass their audit")
    for path, expected in [("group_analysis_native_v1/complete.json", 11599872),
                           ("hiformer_QK_Video_Action_groups_v1/complete.json", 48529408),
                           ("hiformer_TencentGR_10M_Action_groups_v1/complete.json", 153157632)]:
        receipt = read_json(path)
        assert receipt["identity_verified"] and receipt["rows"] == expected
    assert read_json("extended_benchmark_v1/complete.json")["rounds"] == 6
    qk = read_rows("hiformer_QK_Video_Action_groups_v1/group_results.csv")
    taac = read_rows("hiformer_TencentGR_10M_Action_groups_v1/group_results.csv")
    costs = read_rows("extended_benchmark_v1/cost_measurements.csv")
    quality = read_rows("prediction_quality_merrec_v1/paired_deltas.csv")
    old_cells = read_rows("paired_cells.csv")
    new_labels = read_rows("new_training_label_metrics.csv")
    target_labels = read_rows("targeted_training_label_metrics.csv")
    historical = read_rows("paired_labels.csv")
    # The separate historical archive owns the 2028 E references.
    archive = STUDY.parents[1] / "experiments/sisa_paper_report/archive/results/label_results.csv"
    SOURCE_PATHS.add(archive)
    with archive.open(newline="") as stream:
        e_rows = list(csv.DictReader(stream))
    e = {(r["model"], r["dataset"], r["label"]): r for r in e_rows}
    native = {(r["model"], r["dataset"]): r for r in baseline if r["kind"] == "native_baseline"}
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y年%m月%d日 %H:%M")
    text = []
    add = text.append
    add("# RISE 论文补充研究报告\n")
    add("**最终交付范围：14项新增训练全部完成并通过审计，包含追加的Zenith/HyFormer六项跨模型位置复核；正式分组、统计验证、机制观测、预测质量和成本测量均已完成。本文保留全部主要正负结果与未解决的证据边界。**\n")
    add(f"整理时间：{stamp}（北京时间）。对应原稿：More Than a Match: State-Informed Attention for Unified Ranking。\n")
    add("本文统一论文命名RISE与代码命名SISA，AM对应decay，PA对应rotation。本报告包含本次真实执行的研究、完整主要结果、统计与机制定义、反例、计算成本和可复查实验记录。初始可行性文档research_report.md保留，不再作为当前完成状态。\n")
    add("## 技术摘要：已经得到什么结论\n")
    add("本阶段完成14项新训练、三个模型—数据集单元的原生Baseline配对预测、HyFormer–MerRec完整组件分组、用户簇不确定性、AUC正负样本对分解、概率质量检查和受控推理成本测量。14项新训练均经调度状态、退出码、冻结参数、源码、每rank样本覆盖、最终测试双日志及权重实物核查。没有用小样本推理冒充正式评估，也没有改变另一项score_dim研究。\n")
    add("最重要的新增结论有四项：\n\n1. **HiFormer–QK的负面结果与接入位置有关。** 原生宏AUC为0.933438，完整RISE为0.930891；仅保留历史target pooling达到0.934415，仅保留字段交互则为0.929391。这是匹配预算下的单种子位置对照，支持字段位置是本例的重要负面来源；扩展到Zenith后，history_only达到0.935786，高于完整E的0.935221，token_only的0.933800则低于原生0.934635；但HyFormer中完整双阶段0.931308仍最高。证据支持架构相关的位置敏感性，不能据此宣称所有字段attention都不适合RISE。\n2. **RISE收益有明显条件差异。** MerRec原生对照中长历史组的收益通常大于短历史组，但并非单调；空历史区间跨零。TAAC的21–50、51–99历史组改善，1–20组点估计略负且区间跨零。QK的四个非空历史组均下降，follow退化横跨所有长度组。\n3. **累积结构与输入自适应需要分开讨论。** RankMixer–KuaiRand完整RISE优于等可训练参数的局部控制；输入无关的可学习控制却略高于完整RISE。HyFormer–MerRec中完整RISE又优于固定控制约0.010117 AUC。证据支持结构和场景依赖，不能把自适应描述为所有架构的普遍必要条件。\n4. **小参数增量不等于零运行开销，也不等于所有质量指标都改善。** 当前HyFormer实现从Flash Attention转为memory-efficient attention；受控测量存在实际开销。Purchase的AP略降，Offer和Checkout的Logloss/Brier略升，均完整保留。\n")
    add("所有区间均是固定已训练模型下的名义用户簇区间，不是训练seed方差；分组解释和后续对照均属探索性。MerRec历史动作的时点可得性仍存在未解决的数据边界，不能从本研究宣称已证明无泄漏。\n")

    add("## 1. 证据范围、比较关系与训练协议\n")
    add("原稿主表仍保持seed 20262027的4模型×4数据集，共16格；不以诊断后选出的最佳配置回填主表。新增匹配研究使用seed 20262028、两张H100、每卡16384、global batch32768、1 epoch、最大历史100、loader seed2027、辅助初始化seed20260822。已有完整RISE检查点为预先固定的历史参照，新增Baselines及控制从头训练。历史E的完整训练软件清单并未全部保留，因此记录了可核验配置和当前源码，不声称历史环境逐字节相同。\n")
    add("A指独立训练的原生Baseline；B为Content-only；C为AM-only；D为PA-only；E为Full RISE。E−A估计完整方法相对原生骨干的差异，E−B估计保留辅助内容分支之后的组件配置差异，二者不能互换。同模型内部去除局部分数修正的参考分布也不是独立训练的A。\n")
    add("正式评估沿用各自两rank的逐块drop_last覆盖：MerRec 11,599,872行、QK 48,529,408行、TAAC 153,157,632行。所有配对预测按物理part_id、row_offset及标签/元数据逐行一致性核对，先复算全局AUC/Logloss，再进入分组。组内AUC只在正负例均存在时定义；宏平均不悄悄删除不可定义标签。\n")
    rows = []
    for model, ds in [("HyFormer", "MerRec_Action"), ("HiFormer", "QK_Video_Action"), ("HiFormer", "TencentGR_10M_Action")]:
        current = native[model, ds]
        labels = [r for r in e_rows if r["model"] == model and r["dataset"] == ds]
        auc = sum(float(r["sisa_20262028_auc"]) for r in labels) / len(labels)
        ll = sum(float(r["sisa_20262028_logloss"]) for r in labels) / len(labels)
        rows.append([model, ds, number(current["macro_auc"]), number(auc), number(auc-float(current["macro_auc"]), signed=True), number(ll-float(current["macro_logloss"]), signed=True)])
    add(table(["模型", "数据集", "Baseline宏AUC", "RISE宏AUC", "ΔAUC", "ΔLogloss"], rows))
    add("此表按六位逐标签日志重算；后文预测级统计直接使用未舍入预测，因此末尾约10⁻⁷的显示差异是记录精度口径差异。后文AUC差值为绝对单位；乘100才是百分点。\n")

    add("## 2. HiFormer–QK与TAAC：同一方法为何呈现不同条件响应\n")
    for name, data in [("QK-Video", qk), ("TAAC-25", taac)]:
        add(f"### {name}：完整原生配对分组\n")
        add(table(["维度", "组", "N", "Baseline宏AUC", "RISE宏AUC", "ΔAUC", "名义95%区间"],
                  [[r["dimension"], group_name(r["group"]), f"{int(r['rows']):,}", number(r["baseline_auc"]), number(r["rise_auc"]), number(r["delta_auc"], signed=True), interval(r)]
                   for r in data if r["label"].startswith("macro")]))
    add("QK的follow在长度1–20、21–50、51–99、100组的AUC差分别为−0.013424、−0.010743、−0.008366、−0.006546，四组名义区间均在零以下；其正例率约0.080%–0.088%。因此负面结果并非仅集中于一个极小群体，随长度增加其幅度减弱，但没有被长历史完全消除。物品频次最高组占绝大部分QK样本并下降，低频组的少量正向点估计不能覆盖这一总体退化。\n")
    add("TAAC则主要在有一定可见历史的样本上获益：21–50组ΔAUC为+0.023980，51–99组为+0.016602；1–20组为−0.001685且区间跨零，零历史仅1204行，估计不稳定。TAAC没有长度100组，其预处理历史索引上限为99，不能把TAAC较大收益解释为它拥有更长的模型输入。\n")
    add("两数据集的训练目标频次构成差异很大：TAAC约67.08%的测试目标属于训练目标未见物品，QK约0.33%。这提供了一个有内容的群体差异描述，但不是完整原因：在两个数据集中，训练已见和未见组的响应仍不同，标签、划分、筛选规则和token布局也同时变化。频次零不保证该物品从未出现在任何训练历史中，更不能直接称作严格冷启动。\n")
    add("各组Baseline AUC已与增益并列，以避免把组本身更容易预测当成方法效果。逐动作正例数、正例率及完整AUC差异列在附录，不以一个混合的宏正例率代替它们。\n")

    add("## 3. MerRec：完整方法相对原生骨干的已验证条件效果\n")
    add(embedded_note("native_merrec_analysis_verified_zh.md"))
    add("\n## 4. MerRec：AM、PA与辅助内容分支的条件作用\n")
    add(embedded_note("component_analysis_verified_zh.md"))

    add("\n## 5. 新增机制控制：容量、累积、自适应与接入位置\n")
    add("### 5.1 等参数局部控制与固定控制\n")
    add(table(["模型/数据集", "配置", "可训练稠密参数", "宏AUC", "宏Logloss"], [
        ["RankMixer–KuaiRand", "Content-only B", "3,685,895", "0.8543160", "0.1794982"],
        ["RankMixer–KuaiRand", "局部控制L", "3,686,760", "0.8546700", "0.1793947"],
        ["RankMixer–KuaiRand", "固定控制F", "3,685,904", "0.8617022", "0.1743893"],
        ["RankMixer–KuaiRand", "完整RISE E", "3,686,760", "0.8615015", "0.1745697"],
        ["HyFormer–MerRec", "原生Baseline A", "7,706,373", "0.7856338", "0.0855628"],
        ["HyFormer–MerRec", "Content-only B", "7,804,689", "0.7831212", "0.0858484"],
        ["HyFormer–MerRec", "固定控制F", "7,804,797", "0.7907552", "0.0842064"],
        ["HyFormer–MerRec", "完整RISE E", "7,832,349", "0.8008724", "0.0826776"],
    ]))
    add("L保留与E完全相同的活跃参数和初始化，但以token-local控制替代prefix累计；为保证严格局部性，使用固定零居中基准。因而在clipping活跃时，该对照也包含数值稳定化基准的差异，不是无条件纯scan开关。RankMixer只有一个query-reference target pooling位置，避免了在sequence-end位置伪称局部但仍读取全局终点的问题。完整E相对L提高0.0068315 AUC，支持这套累积构造相对所选等参数局部替代的价值，不能推广为所有等容量替代均无效。\n")
    add("F保留原prefix、reference、minimax和clipping，改用输入无关但可学习的head decay与phase增量。F不是等参数控制，其相位初始为零；容量和初始化差异仍在。RankMixer中F比E高0.0002007，不能宣称统计等价或输入自适应普遍必要。HyFormer中E比F高0.0101172，说明不同架构/场景的响应明显不同。既有RankMixer AM-only的0.8617185同样高于E，全部结果应一起保留，而不选择有利配置。\n")
    add("### 5.2 HiFormer–QK接入位置对照\n")
    add(table(["配置", "活跃RISE位置", "可训练稠密参数", "宏AUC", "ΔAUC vs Baseline", "宏Logloss"], [
        ["Baseline A", "0", "27,824,132", "0.93343800", "0", "0.07563200"],
        ["完整E", "1历史汇聚+3字段层", "27,889,090", "0.93089125", "−0.00254675", "0.07584525"],
        ["target_only", "1历史汇聚", "27,826,102", "0.93441475", "+0.00097675", "0.07478300"],
        ["field_only", "3字段层", "27,887,120", "0.92939075", "−0.00404725", "0.07632900"],
    ]))
    add("相同训练预算下，仅历史汇聚改善AUC和Logloss，仅字段交互则使两者变差；完整模型位于两者之间。这支持字段接入是该负面场景的重要来源。它不唯一识别衰减、旋转、字段顺序或优化适应中的哪一项导致退化，也不是等参数位置比较。保留位置的初始化和seed offsets逐项一致，禁用位置是真正的None，而不是不参与前向的占位参数。\n")
    add("这些追加控制是根据先前观察形成的探索性后续研究，不是独立预注册确认。主表仍保留原始完整RISE负例；不把事后选择的target_only替换回原seed20262027主表。\n")

    add("### 5.3 跨模型验证：Zenith与HyFormer\n")
    add("在HiFormer–QK的位置结果形成后，按用户要求扩展到另外两个正式骨干，每个补一个原生Baseline和两个单位置控制，共6项新训练。数据、seed、训练预算、每rank覆盖和保留位置的初始化均对齐；历史完整E继续使用冻结参照。结果如下，差值为绝对AUC单位。\n")
    add(table(["模型", "配置", "Baseline宏AUC", "完整E宏AUC", "该位置宏AUC", "ΔAUC vs Baseline", "ΔAUC vs E"],
              [[r["model"], r["mode"], number(r["Baseline_AUC"],8), number(r["Full_RISE_AUC"],8), number(r["Placement_AUC"],8), number(r["delta_AUC_vs_Baseline"],8,True), number(r["delta_AUC_vs_Full_RISE"],8,True)] for r in site_comparisons if r["label"] == "__macro__"]))
    add("**Zenith复现了位置排序，但没有复现完整模型退化。** history_only保留1个历史target pooling，token_only保留3个统一token交互层；前者比原生提高0.00115100，后者比原生下降0.00083475。历史位置单独接入还比完整E提高0.00056525，四个动作相对原生的AUC均提升；token_only四个动作均下降。但history_only的like低于完整E，且其宏Logloss为0.07440950，略差于完整E的0.07438775，不能称作所有任务或指标均优。与此同时，Zenith完整E仍比原生提高0.00058575。因此，共同现象是位置选择能影响收益，不是完整RISE在QK对所有模型有害。\n")
    add("**HyFormer给出了必要的结构反例。** sequence_only保留3个序列编码位置，query_only保留3个query解码位置；二者宏AUC相对原生分别提高0.00007800和0.00036325，完整双阶段提高0.00045000。完整E相对query_only仅高0.00008675；两种单位置变体稠密参数数量相同，但计算角色不同。该模型两个位置都在处理行为序列信息，不能把sequence_end参考方式直接等同于非时序字段位置。sequence/query单独控制的follow和share仍有负向点估计，宏平均的小幅正向不能解释为每个任务改善。\n")
    add("因此，可写入论文的结论是：RISE的位置敏感性跨架构存在，而有利插入点取决于该层token的语义与信息流。HiFormer和Zenith的结果与历史汇聚优于单独字段/token混合这一假设一致；HyFormer说明序列编码和query解码可以提供互补收益。这里的互补仅指已测配置的性能排序，不能解释为已证明AM与PA统计交互或超加性。位置控制改变活跃参数量，全部为单训练seed的探索性后续实验，没有据此声明显著性或普遍规律，也没有对TAAC做位置实验。\n")
    add(table(["模型", "设置", "活跃稠密参数", "宏Logloss", "任务号"],
              [[r["model"],r["mode"],f"{int(r['dense_parameters']):,}",number(r["macro_logloss"],8),r["job_task_id"]] for r in replication]))
    add("Zenith与HyFormer的六项日志、冻结配置、两rank接入点与seed、最终Test双日志、checkpoint实物及调度状态均通过。QK每项训练保留394,428,416行，验证50,036,736行，测试48,529,408行，与对应历史E的逐rank覆盖一致。这里证明的是记录中的覆盖一致，历史软件版本与数据字节完全相同未被追溯证明。初始array并发上限2已按用户要求在调度器中移除，源码冻结文件保持不变；运行回执记录这次调度调整。\n")

    add("## 6. 输出不变的compiled机制观测\n")
    add(embedded_note("compiled_mechanism_v1_interpretation.md").replace("发生显著的选择再分配", "发生幅度较大的选择再分配"))
    add("\n### 数值失败记录与拒绝采用的结果\n")
    add("643858的eager观测不能通过compiled预测门槛。643882的四路对照表明，compiled capture/plain及eager hook/plain各自逐位一致，跨模式仍不同；这不能被简单归咎于hook修改输出。643911在B完整输出后发生SIGSEGV，B子阶段通过而全job失败；仅E在643949独立补齐并通过。643918中E的控制重算最大Δ误差0.3125，最大selection TV误差0.026538，故拒绝其AM/PA状态解释。644002直接暴露_final_channels中间量也改变了compiled预测，故同样拒绝。没有为得到好看的机制图放宽门槛，也没有把无法忠实读出的内部量写成原执行事实。\n")

    add("## 7. 受控推理成本：必须区分原生开销和组件开销\n")
    cost_table = []
    for batch in [64, 256, 512]:
        for order in ["forward01", "reverse01"]:
            group = {r["variant"]: r for r in costs if int(r["batch"]) == batch and r["order"] == order}
            cost_table.append([batch, "A→B→E" if order == "forward01" else "E→B→A", *[number(group[v]["median_ms"], 3) for v in "ABE"], number(group["E"]["E_over_A"], 3), number(group["E"]["E_over_B"], 3)])
    add(table(["batch", "执行顺序", "Baseline A / ms", "Content-only B / ms", "RISE E / ms", "E/A", "E/B"], cost_table))
    add("一次H100 80GB分配、相同GPU UUID，BF16与torch.compile，使用同一固定真实输入缓存。64/256/512为嵌套的真实样本前缀，未复制样本扩batch。每variant独立进程，先1000次warmup，再10个block×100次forward；表中是block平均耗时的中位数。排除权重加载、编译、H2D和数据读取；不是端到端服务延迟或请求P95。正反顺序全部保留，没有选择最优一轮。\n")
    add("E/A实测范围约1.266–1.577；这是当前实现和这些输入上的开销，不能从E/B较小就宣称相对原生几乎零开销。profiler在A观察到Flash Attention，在B/E观察到memory-efficient attention，说明额外运算与后端选择共同进入测量。该观察不独立量化两者的因果贡献。\n")
    mem = []
    for batch in [64, 256, 512]:
        for v in "ABE":
            r = next(r for r in costs if int(r["batch"]) == batch and r["variant"] == v and r["order"] == "forward01")
            mem.append([batch, v, number(r["resident_MiB"], 2), number(r["peak_allocated_MiB"], 2), number(r["incremental_peak_MiB"], 2), r["kernel"]])
    add(table(["batch", "配置", "warmup后allocated/MiB", "峰值allocated/MiB", "峰值增量/MiB", "SDPA内核"], mem))
    add("增量定义为峰值减warmup后allocated，不冒称包含所有持久缓存的纯activation内存。总峰值与增量同时给出，避免大embedding常驻内存掩盖临时分配差异。完整两轮内存、时钟/功耗、kernel与原始block值均保留。\n")

    add("## 8. ROC-AUC以外的预测质量\n")
    add(table(["任务", "正例率", "ΔAUC", "Baseline AP", "RISE AP", "ΔAP", "ΔLogloss", "ΔBrier"],
              [[r["label"], "逐任务定义" if not r["positive_rate"] else f"{float(r['positive_rate'])*100:.5f}%", number(r["delta_ROC_AUC"], signed=True), number(r["Baseline_AP"]), number(r["RISE_AP"]), number(r["delta_AP"], 8, True), number(r["delta_Logloss"], 8, True), number(r["delta_Brier"], 9, True)] for r in quality]))
    add("AP采用sklearn average_precision_score，明确不同于PR曲线梯形积分。宏AP从0.07790750到0.09280871，但Purchase的AP略降0.00007766；这只是反向点估计，没有宣称其统计显著。Offer和Checkout的AUC提升，同时Logloss和Brier略增。因此，排序改善不保证每个反馈的概率质量或每个阈值下的精度都改善。\n")
    add("校准检查使用所有模型和任务共享的预设20个概率区间，保留空bin；200行分箱数据的N和正例数与全体样本严格对账。宏ECE几乎不变且四个标签上升。ECE依赖所选网格，不是无条件校准判定，也没有用测试集选择阈值或拟合校准器。完整逐bin表保存在校准CSV。\n")

    add("## 9. 数据构造审计与外推边界\n")
    semantics = embedded_note("data_semantics_audit.md")
    semantics = semantics.replace("真实全量脚本尚未执行；正式完成需核对频次总和、8块输出总数、每块row-key散列、与四变体预测的ID/标签匹配及最终资源记录。", "最终状态：真实全量构建与人口统计已完成，144,108,235条训练目标频次和11,599,872条正式测试旁表已核对；八块row-key、标签及四变体样本身份检查通过。阶段测试数只描述原始实现时的定向测试，最终测试回执另列。")
    semantics = semantics.replace("压缩大小需实际执行后记录。", "此处是原始资源估算，不作实测压缩大小报告。")
    semantics = semantics.replace("预计数GB量级，尚未在真实完整块上实测。", "最初按数GB量级估算；该估算不应作为实测峰值。")
    semantics = semantics.replace("应从首块日志和调度器资源记录获得真实成本，再决定并行度。", "正式全量作业已完成，本节保留算法资源推导，不将调度预算冒称实测峰值。")
    add(semantics)
    add("\n上述数据语义笔记保留了实现时的阶段记录；截至最终汇总，后续CPU全量频次、特征旁表、人口统计与预测配对均已完成。原始事件级数据仍未在已查本地范围找到，真实未来信息影响比例依然未知。\n")
    add("## 10. 论文写作建议与不能越过的结论\n")
    add("建议正文以原生A/E的条件效果为主，再用B/C/D/E解释累积调制；增加三个模型的位置对照，说明HiFormer–QK例外及其跨架构边界；用RankMixer局部/固定控制与HyFormer固定控制呈现机制依赖；将完整组表、AUC分解、数据审计、编译数值验证和成本原始轮次放入附录。不要把所有探索性结果塞入一个总体平均提升。\n")
    add("可以写：在既定离线协议下，RISE的效果依赖样本条件、架构及插入点；所测等参数局部控制不足以达到完整累积模型，而固定控制在两个场景中的相对表现不同。不能写：所有长历史/稀疏样本都受益、动态控制处处必要、AM/PA有普遍超加性协同、attention图恢复真实兴趣、已证明时点无泄漏、已经训练至收敛、低维意味着几乎零推理开销、用户簇区间证明训练种子稳定。\n")
    add("分组边界与组间比较的选择时点严格区分：长度边界和训练频次规则在正式效果读取前固定；三项直接组间比较是在初次E−B分组均值检查后选出，原生研究沿用同样定义；Offer比较为进一步追加。所有这些比较均为探索性名义区间，未作多重比较校正。\n")
    add("## 11. 十四项新训练的完整完成记录\n")
    add(table(["任务", "模型/数据集", "设置", "宏AUC", "宏Logloss", "状态", "用时"],
              [[r["job_task_id"], r["model"]+"/"+r["dataset"], r.get("setting", r.get("mode")), number(r["macro_auc"], 7), number(r["macro_logloss"], 7), "COMPLETED / 0:0；审计通过", r["elapsed"]] for r in baseline+targeted+replication]))
    add("训练完成不单凭提交或进度条判定：对十四项运行分别核对scheduler、outcome、Params、源码散列、每rank三split覆盖、双来源最终Test、标签和参数量。位置实验还检查两rank构造记录中的6/1/3个实际接入点及种子偏移。所有失败的诊断与被取消的错误CPU入口保留，不混入成功结果。\n")
    add("### 非训练作业与失败尝试\n")
    add(table(["任务号", "用途", "结果/边界"], [
        ["643821", "全训练频次与MerRec分组特征", "完成；144,108,235训练目标，11,599,872测试旁表"],
        ["643834", "GPU smoke", "完成；仅小样本验证，不是正式结果"],
        ["643839", "B/C/D/E正式预测", "完成；每模型11,599,872行，40指标与历史差<5×10⁻⁷"],
        ["643879 / 643880 / 643881", "三个原生配对的正式预测", "完成；全局指标与样本身份通过"],
        ["643859 → 643871", "人口统计", "首次输出mtime漂移失败，保留证据并独立重算后成功"],
        ["643869 / 643917", "组件/原生分组", "完成；保留不可定义与小支持组"],
        ["643902_0 / _1", "QK/TAAC训练频次", "完成；394,647,318 / 586,117,494训练目标"],
        ["644033 / 644034", "HiFormer分析旧入口", "QK因输出目录约束失败；取消仍待运行的同错误TAAC入口"],
        ["644047 / 644048", "修正入口后的HiFormer分析", "完成；分别48,529,408 / 153,157,632行"],
        ["643909 / 644019", "组件/原生AUC结构", "完成；精确组对分解与同用户协方差"],
        ["643956", "AP/Brier/校准", "完成；同样本无校准器拟合"],
        ["643858 / 643882", "机制初试与四路诊断", "eager数值门槛失败；定位到跨执行路径差异"],
        ["643911 / 643949", "完整compiled观测", "前者B子阶段成功后SIGSEGV；仅补E成功，合并6144条验收记录"],
        ["643918 / 644002", "AM/PA重算与内部状态暴露", "E未通过预测/分数一致性，拒绝作为机制证据"],
        ["643870 / 643950", "初版/扩展成本", "扩展完成6轮18模型；初版单独留档，不混合协议"],
    ]))

    add("## 附录A. HiFormer两数据集的完整逐动作分组\n")
    for title, data in [("QK-Video", qk), ("TAAC-25", taac)]:
        add(f"### {title}\n")
        add(table(["维度/组", "动作", "N", "正例数", "正例率", "Baseline AUC", "RISE AUC", "ΔAUC", "名义95%区间"],
                  [[r["dimension"]+"/"+group_name(r["group"]), r["label"], f"{int(r['rows']):,}", r["positives"], f"{float(r['positive_fraction'])*100:.5f}%", number(r["baseline_auc"]), number(r["rise_auc"]), number(r["delta_auc"], signed=True), interval(r)] for r in data if not r["label"].startswith("macro")]))
    add("## 附录B. 新增训练的逐动作指标\n")
    add(table(["任务", "模型/数据集", "设置", "动作", "AUC", "Logloss"],
              [[r["job_task_id"], r["model"]+"/"+r["dataset"], r.get("setting",r.get("mode")), r["label"], number(r["AUC"]), number(r["logloss"])] for r in new_labels+target_labels+site_labels]))
    add("### Zenith与HyFormer逐动作位置比较\n")
    add(table(["模型", "位置", "动作", "Baseline AUC", "完整E AUC", "单位置AUC", "ΔAUC vs Baseline", "ΔAUC vs E"],
              [[r["model"], r["mode"], r["label"], number(r["Baseline_AUC"]), number(r["Full_RISE_AUC"]), number(r["Placement_AUC"]), number(r["delta_AUC_vs_Baseline"],6,True), number(r["delta_AUC_vs_Full_RISE"],6,True)] for r in site_comparisons if r["label"] != "__macro__"]))
    add("## 附录C. 保持不变的原seed20262027主表\n")
    add(table(["模型", "数据集", "Baseline AUC", "RISE AUC", "ΔAUC"],
              [[r["model"],r["dataset"],number(r["baseline_AUC"]),number(r["rise_AUC"]),number(r["delta_AUC"],signed=True)] for r in old_cells]))
    add("## 附录D. 复现与归档\n")
    add("所有本阶段文件在experiments/rise_sample_analysis；大体量预测、模型、输入缓存、临时编译文件和运行清单在artifacts及checkpoints/logs。未修改核心训练代码、根训练依赖锁文件或其他agent的维度消融。绘图使用独立figure_env的uv锁定环境。原计划、旧稿及失败尝试保留，不将已归档手稿覆盖成新的推断。\n")
    add("```bash\n.venv/bin/python experiments/rise_sample_analysis/scripts/collect_new_experiments.py\n.venv/bin/python experiments/rise_sample_analysis/scripts/collect_targeted_controls.py\n.venv/bin/python experiments/rise_sample_analysis/scripts/collect_site_replication.py\n.venv/bin/python experiments/rise_sample_analysis/scripts/run_study_tests.py\nuv run --project experiments/rise_sample_analysis/figure_env --locked python experiments/rise_sample_analysis/scripts/plot_component_analysis.py\nuv run --project experiments/rise_sample_analysis/figure_env --locked python experiments/rise_sample_analysis/scripts/plot_native_merrec_analysis.py\n.venv/bin/python experiments/rise_sample_analysis/scripts/build_final_dossier.py\n```\n")
    add("本研究测试按模块独立进程执行且不跳过测试。合并进程的unittest发现曾在PyTorch编译状态测试处SIGSEGV，故不声称合并进程全套通过；失败日志保留在artifacts/rise_sample_analysis/validation。生产训练和诊断采用独立进程，独立测试回执记录了每个模块的实际退出状态。\n")
    add("统计参考：DeLong、DeLong、Clarke-Pearson，Biometrics 1988，44(3):837–845，DOI 10.2307/2531595；Obuchowski，Biometrics 1997，53(2):567–578，DOI 10.2307/2533958。本研究公开了具体影响函数、用户聚合和bootstrap数值验证，不冒称完全复刻外部软件或其所有有限样本修正。\n")
    report = STUDY / "final_research_report.md"
    report.write_text("\n".join(text) + "\n")
    SOURCE_PATHS.add(Path(__file__).resolve())
    manifest = {"generated_at": stamp, "training_runs_audited": 14,
                "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "sources": [{"path": str(p.relative_to(STUDY.parents[1])), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(SOURCE_PATHS)],
                "status": "built; requires final document and figure QA"}
    (RESULTS / "final_dossier_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"{report}: {len(report.read_text().splitlines())} lines; {len(report.read_text())} characters")


if __name__ == "__main__":
    main()
