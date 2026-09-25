#!/usr/bin/env python3
"""Build the standalone Chinese dimension report from fully audited evidence."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "experiments/ablations/sisa_score_dim"
LABELS = ("Like", "Cart", "Offer", "Checkout", "Purchase")
DIMENSIONS = (8, 16, 32, 64)


def independent_metrics(path: Path) -> dict:
    text = path.read_text(errors="replace" if path.suffix == ".out" else "strict")
    sections = re.split(r"\*+ Test evaluation \*+", text)
    if len(sections) != 2:
        raise ValueError(f"Expected exactly one final Test section: {path}")
    values = re.findall(r"\[Task:\s*([^\]]+)\]\[Metrics\]\s*"
                        r"logloss:\s*([^\s]+)\s+-\s+AUC:\s*([^\s]+)", sections[1])
    if len(values) != 5 or {label for label, _, _ in values} != set(LABELS):
        raise ValueError(f"Invalid final label coverage: {path}")
    metrics = {label: {"AUC": Decimal(auc), "logloss": Decimal(loss)}
               for label, loss, auc in values}
    for row in metrics.values():
        if not all(value.is_finite() for value in row.values()):
            raise ValueError(f"Nonfinite metric: {path}")
        if not 0 <= row["AUC"] <= 1 or row["logloss"] < 0:
            raise ValueError(f"Invalid metric range: {path}")
    return metrics


def verify(evidence: dict) -> tuple[dict, dict]:
    if (evidence.get("model"), evidence.get("dataset"), evidence.get("seed")) != ("HyFormer", "MerRec_Action", 20262028):
        raise ValueError("Evidence is outside the authorized model, dataset or seed")
    rows = evidence["runs"]
    if not evidence.get("complete") or len(rows) != 6:
        raise ValueError("Final report requires six audited runs, including all three r16 seeds")
    expected_keys = {(16, seed) for seed in (20262027, 20262028, 20262029)} | {
        (dim, 20262028) for dim in (8, 32, 64)}
    if len({row["job_task_id"] for row in rows}) != 6 or {(row["score_dim"], row["seed"]) for row in rows} != expected_keys:
        raise ValueError("Run/seed coverage differs from the authorized reporting basis")
    reference = next(row for row in rows if row["score_dim"] == 16 and row["seed"] == 20262028)
    cells = {row["score_dim"]: row for row in evidence["summary"]}
    if set(cells) != set(DIMENSIONS) or len(evidence["summary"]) != 4:
        raise ValueError("Four reported dimension points are required")
    checks = []
    parsed_runs = {}
    for row in rows:
        dim = row["score_dim"]
        if not row["eligible"] or row["state"] != "COMPLETED" or row["exit_code"] != "0:0" or row["issues"]:
            raise ValueError(f"Unaccepted evidence at r={dim}")
        parsed = independent_metrics(ROOT / row["checkpoint_log"])
        parsed_runs[row["job_task_id"]] = parsed
        if parsed != independent_metrics(ROOT / row["slurm_log"]):
            raise ValueError(f"Raw log disagreement at r={dim}")
        for label in LABELS:
            for metric in ("AUC", "logloss"):
                if abs(parsed[label][metric] - Decimal(str(row["metrics"][label][metric]))) > Decimal("1e-12"):
                    raise ValueError(f"Saved metric disagrees with raw Test: {dim}/{label}/{metric}")
        for metric, key in (("AUC", "macro_auc"), ("logloss", "macro_logloss")):
            macro = sum(parsed[label][metric] for label in LABELS) / Decimal(5)
            if abs(macro - Decimal(str(row[key]))) > Decimal("1e-12"):
                raise ValueError(f"Macro arithmetic mismatch: {dim}/{metric}")
        is_four_gpu = dim == 64 or (dim == 16 and row["seed"] == 20262027)
        coverage_file = "data_coverage_ws4.csv" if is_four_gpu else "data_coverage.csv"
        with (DIRECTORY / "configs" / coverage_file).open() as stream:
            expected_coverage = [{key: value if key == "split" else int(value)
                                  for key, value in record.items() if key != "status"}
                                 for record in csv.DictReader(stream)]
        if row["coverage"] != expected_coverage:
            raise ValueError(f"Per-rank sample coverage differs from its frozen layout at r={dim}")
        world_size, per_gpu_batch = (4, 8192) if is_four_gpu else (2, 16384)
        if (row.get("world_size"), row.get("per_gpu_batch"), row.get("global_batch")) != (world_size, per_gpu_batch, 32768):
            raise ValueError(f"Incorrect global-batch layout: {dim}")
        if row.get("gradient_checkpointing") is not False:
            raise ValueError(f"Execution adjustment not explicitly disclosed: {dim}")
        scope = "dimension_with_four_gpu_layout" if dim == 64 else "dimension_only"
        if dim != 16 and row.get("comparison_scope") != scope:
            raise ValueError(f"Wrong comparison scope: {dim}")
        expected_total = reference["total_parameters"] + 7680 * (dim - 16)
        expected_dense = reference["dense_parameters"] + 7680 * (dim - 16)
        if row["total_parameters"] != expected_total or row["dense_parameters"] != expected_dense:
            raise ValueError(f"Parameter count mismatch: {dim}")
        for field in ("checkpoint_log", "slurm_log"):
            path = ROOT / row[field]
            checks.append(dict(score_dim=dim, seed=row["seed"], path=row[field],
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for dim, summary in cells.items():
        group = sorted((row for row in rows if row["score_dim"] == dim), key=lambda row: row["seed"])
        expected_seeds = [20262027, 20262028, 20262029] if dim == 16 else [20262028]
        if (not summary["eligible"] or summary["n_runs"] != len(expected_seeds)
                or summary["seeds"] != expected_seeds
                or summary["job_task_ids"] != [row["job_task_id"] for row in group]
                or "job_task_id" in summary):
            raise ValueError(f"Summary must retain the true run count and provenance at r={dim}")
        for metric, key in (("AUC", "macro_auc"), ("logloss", "macro_logloss")):
            label_means = {label: sum(parsed_runs[row["job_task_id"]][label][metric] for row in group) / len(group)
                           for label in LABELS}
            expected_macro = sum(label_means.values()) / 5
            if abs(expected_macro - Decimal(str(summary[key]))) > Decimal("1e-12"):
                raise ValueError(f"Reported mean mismatch: {dim}/{metric}")
            for label, value in label_means.items():
                if abs(value - Decimal(str(summary["metrics"][label][metric]))) > Decimal("1e-12"):
                    raise ValueError(f"Reported label mean mismatch: {dim}/{label}/{metric}")
    history = evidence.get("attempt_history", [])
    if (len(history) != 2 or {row["job_task_id"] for row in history} != {"643736_2", "643901_2"}
            or any(row["eligible"] for row in history)):
        raise ValueError("Both failed r64 attempts must remain separately visible")
    validation = dict(status="share_with_caveats", dimensions=4, labels_per_dimension=5,
                      underlying_runs=6, independently_verified_label_metric_values=60,
                      reported_label_metric_values=40,
                      macro_calculation="Exact Decimal mean across five Test labels, then across all three runs for r16",
                      comparison_groups={"three_run_mean_reference": [16], "single_run_variants": [8, 32, 64]},
                      issues=[], caveats=["r16 averages three runs; other dimensions have only one run each",
                          "r16 references have mixed GPU layouts and internal seed overrides",
                          "r64 uses four GPUs at batch 8192; r8/r32 use two GPUs at batch 16384",
                          "same global batch does not make DDP sharding and drop_last coverage identical",
                          "historical full software and source snapshots are unavailable",
                          "coverage counts do not prove historical sample byte identity"],
                      sources=checks)
    return cells, validation


def table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |",
                      "| " + " | ".join(["---"] * len(header)) + " |",
                      *["| " + " | ".join(row) + " |" for row in rows]])


def report_text(evidence: dict, cells: dict) -> str:
    reference = cells[16]
    reference_runs = [row for row in evidence["runs"] if row["score_dim"] == 16]
    variants = {row["score_dim"]: row for row in evidence["runs"] if row["score_dim"] != 16}
    protocol_reference = next(row for row in reference_runs if row["seed"] == 20262028)
    seed_table = table(["实验 seed", "任务", "宏平均 AUC", "宏平均 LogLoss", "GPU 数×单卡 batch"],
                       [[str(row["seed"]), row["job_task_id"], f'{row["macro_auc"]:.7f}',
                         f'{row["macro_logloss"]:.7f}', f'{row["world_size"]}×{row["per_gpu_batch"]:,}']
                        for row in reference_runs])
    summary_rows = []
    for dim in DIMENSIONS:
        row = cells[dim]
        layout = ("4×8,192（1次）；2×16,384（2次）" if dim == 16 else
                  f'{variants[dim]["world_size"]}×{variants[dim]["per_gpu_batch"]:,}')
        summary_rows.append([str(dim), "三次均值" if dim == 16 else "单次", f'{row["macro_auc"]:.7f}', f'{row["macro_logloss"]:.7f}',
                             f'{row["macro_auc"] - reference["macro_auc"]:+.7f}',
                             f'{row["macro_logloss"] - reference["macro_logloss"]:+.7f}',
                             layout])
    summary = table(["每头维度", "统计口径", "宏平均 AUC ↑", "宏平均 LogLoss ↓", "ΔAUC", "ΔLogLoss", "GPU 数×单卡 batch"], summary_rows)
    auc = table(["任务", "8 维（单次）", "16 维（三次均值）", "32 维（单次）", "64 维（单次、四卡）"],
                [[label, *[f'{cells[dim]["metrics"][label]["AUC"]:.7f}' for dim in DIMENSIONS]] for label in LABELS])
    loss = table(["任务", "8 维（单次）", "16 维（三次均值）", "32 维（单次）", "64 维（单次、四卡）"],
                 [[label, *[f'{cells[dim]["metrics"][label]["logloss"]:.7f}' for dim in DIMENSIONS]] for label in LABELS])
    label_changes = []
    for dim in (8, 32, 64):
        auc_wins = sum(cells[dim]["metrics"][label]["AUC"] > reference["metrics"][label]["AUC"] for label in LABELS)
        loss_wins = sum(cells[dim]["metrics"][label]["logloss"] < reference["metrics"][label]["logloss"] for label in LABELS)
        label_changes.append(f"- {dim} 维单次结果相对 16 维三次均值：{auc_wins}/5 个标签 AUC 更高，{loss_wins}/5 个标签 LogLoss 更低。")
    params = table(["每头维度", "旋转平面/头", "SISA 分支参数", "全模型可训练参数", "稠密可训练参数"],
                   [[str(dim), str(dim // 2), f"{7680 * dim + 3096:,}",
                     f'{cells[dim]["total_parameters"]:,}', f'{cells[dim]["dense_parameters"]:,}'] for dim in DIMENSIONS])
    coverage_rows = []
    for split in ("train", "valid", "test"):
        before = sum(row["processed_rows"] for row in protocol_reference["coverage"] if row["split"] == split)
        after = sum(row["processed_rows"] for row in variants[64]["coverage"] if row["split"] == split)
        coverage_rows.append([split, f"{before:,}", f"{after:,}", f"{after-before:+,}",
                              f"{(after-before)/before*100:+.4f}%"])
    coverage = table(["划分", "两卡协议保留行数", "四卡协议保留行数", "差值", "相对变化"], coverage_rows)
    attempts = table(["维度", "seed", "任务", "尝试", "状态", "处理"],
                     [[str(row["score_dim"]), str(row["seed"]), row["job_task_id"], str(row["attempt"]),
                       "COMPLETED / 0:0", "历史参考，组成三次均值" if row["score_dim"] == 16 else
                       ("两卡原设置" if row["score_dim"] != 64 else "四卡，单卡 batch 减半")]
                      for row in evidence["runs"]]
                     + [["64", "20262028", "643736_2", "1", "FAILED / 1:0", "两卡原设置首批 OOM"],
                        ["64", "20262028", "643901_2", "2", "FAILED / 1:0", "两卡外层检查点仍在首批 OOM"]])
    best_auc = max(DIMENSIONS, key=lambda dim: cells[dim]["macro_auc"])
    best_loss = min(DIMENSIONS, key=lambda dim: cells[dim]["macro_logloss"])
    delta64 = cells[64]["macro_auc"] - reference["macro_auc"]
    loss_delta64 = cells[64]["macro_logloss"] - reference["macro_logloss"]
    offer_delta64 = cells[64]["metrics"]["Offer"]["AUC"] - reference["metrics"]["Offer"]["AUC"]
    checkout_loss_delta = cells[64]["metrics"]["Checkout"]["logloss"] - reference["metrics"]["Checkout"]["logloss"]
    with (DIRECTORY / "notes/runtime_artifact_inventory.csv").open() as stream:
        artifacts = list(csv.DictReader(stream))
    model_files = sum(row["path"].endswith(".model") for row in artifacts)
    return f"""# HyFormer–MerRec SISA 投影维度消融报告

## 结果摘要

本轮覆盖每头 8、16、32、64 维，五个行为标签均已核验。
16 维按原主实验三次运行（seed 20262027、20262028、20262029）的等权均值展示；
8、32、64 维各为 seed 20262028 的单次结果。四个展示点由六个实际运行组成。
按这一口径，宏平均 AUC 最高的是 {best_auc} 维，LogLoss 最低的是 {best_loss} 维。
64 维的宏平均 AUC 相对 16 维三次均值为 {delta64:+.7f}，LogLoss 差值为 {loss_delta64:+.7f}。
64 维只有 Offer 的 AUC 高于 16 维三次均值，其余四个标签均低于该均值。
64 维的两次两卡尝试因显存不足失败，最终采用四卡、每卡 batch 8,192，
全局 batch 仍为 32,768，使用原模型和编译设置完成重试。
三次参考本身混合了四卡和两卡布局，其他维度的重复次数也少于参考，
因此主表是不同重复次数、带协议差异的描述性比较，不能将差值完全归因于维度。

报告口径修订截至 {evidence['generated_at']}，范围仅为 HyFormer–MerRec。
先前报告误将 seed 20262028 的单次 16 维数值作为展示对照；该单次值只保留用于
实际训练配置的追溯，不再代替三次均值。此次更正没有重新训练模型。
各维度没有完成统一的三次重复，不支持跨维度统计显著性或普遍最优结论。

## 16 维三次运行及平均方式

{seed_table}

对每个运行先等权平均五个标签，再等权平均三个运行。AUC 为
`(0.8061308 + 0.8008724 + 0.8065828) / 3 = 0.804528666666...`；LogLoss 为
`(0.0812370 + 0.0826776 + 0.0820006) / 3 = 0.081971733333...`。
逐标签的 16 维结果也采用三次均值；不按各运行样本数加权，不把三次预测拼接后重新计算 AUC。
本次从三次运行的 checkpoint 与 Slurm 最终 Test 段重新提取，并与已有主实验三次汇总逐项核对。

## 维度含义与评价口径

研究变量为 `sisa_score_dim`，即每个注意力头的 B、C 投影通道数。
完整 SISA 在这些通道上执行累计衰减和二维成对旋转，形成 softmax 之前的附加分数。
骨干 token_dim=256、num_heads=2，原有每头 Q/K 维度为 128；本轮保持头数不变。
单个 B 或 C 投影在两个头上的总宽度为 2r，每头有 r/2 个旋转平面。
三个序列注意力和三个 query-to-sequence 注意力接入点统一改变 r。

五个标签为 Like、Cart、Offer、Checkout、Purchase。宏平均为五个标签各自指标的
等权算术平均，不是把五个标签预测拼起来重新计算的 AUC，也不是按标签阳性数加权。
AUC 越高越好，LogLoss 越低越好；主表 Δ 均为当前维度减去 16 维三次运行均值，单位为指标原值。
所有指标来自最终 Test evaluation，checkpoint 按原验证集宏平均 AUC 规则选择。
所有预设维度及三次历史参考均完整展示。16 维的启动协议参考仍为 seed 20262028，
但报告展示参考改为完整三次均值，二者含义明确区分。
宏平均由日志中保留六位小数的标签指标重新计算，表内七位小数是算术展示精度，
不代表新增的测量精度或统计显著性。

## 宏平均结果与执行差异

{summary}

8 维和 32 维单次结果的两项宏平均指标均未超过 16 维三次均值。
64 维 AUC 略低于该均值，LogLoss 更低；先前“64 维宏平均 AUC 最好”的表述不再成立。
64 维行保留真实结果，同时标明四卡布局；其测试保留样本数较两卡协议增加 40,960 行。

## 各行为任务的变化并不一致

### AUC

{auc}

### LogLoss

{loss}

{chr(10).join(label_changes)}

32、64 维均只有 Offer 的 AUC 高于 16 维三次均值，其余四个标签均更低。
64 维的 Offer AUC 差值为 {offer_delta64:+.7f}，对宏平均贡献 {offer_delta64/5:+.7f}；
其余四个标签合计贡献 {delta64-offer_delta64/5:+.7f}，宏平均净差为 {delta64:+.7f}。
按修正后的参考，64 维 LogLoss 改善贡献最大的标签是 Checkout，标签差值为
{checkout_loss_delta:+.7f}，对宏平均贡献 {checkout_loss_delta/5:+.7f}；先前以 Like 为主的归因也随之更正。
8 维的五个标签 AUC 均低于 16 维三次均值。宏平均与逐标签结果应同时报告，
不能用单个提升标签代替本轮整体结论。

## 参数量随维度增加，不能解释为等容量实验

{params}

单处 SISA 的可训练参数量为 `(5/2)×H×r×D + H×D + 2×H`，本配置六处合计为
`7680×r + 3096`。表中 SISA 分支、全模型和稠密参数是不同口径；全模型包含大规模嵌入参数。
维度变化同时改变内容通道、旋转平面数量、参数量和初始化分布。
现有实现未额外按 r 加入 `1/sqrt(r)` 分数归一化，因此也不能把结果解释为
已隔离所有尺度因素的纯维度效应。

## 固定训练协议与实际数据覆盖

- 模型及数据：HyFormer、MerRec_Action；完整 SISA，保留衰减、旋转、接入点和掩码处理。
- 8、32 维：两卡 DDP，每卡 batch 16,384；64 维：四卡 DDP，每卡 batch 8,192。
- 16 维参考：seed 20262027 为四卡×8,192，seed 20262028、20262029 为两卡×16,384。
- 所有最终设置的名义全局 batch 均为 32,768，梯度累积均为 1。
- 训练 1 epoch，max_len=100，token_dim=256，num_heads=2，num_layers=3。
- 新增 8、32、64 维：实验 seed=20262028，各 rank 使用 seed+rank；loader seed=2027；SISA 参数 seed=20260822。
- 16 维 seed 20262029 的 loader seed=2028、SISA 参数 seed=20260823；seed 20262027 日志未显式记录这两项覆盖。
- AdamW 稠密学习率 0.0001、weight_decay=0；Adagrad 稀疏学习率 0.05。
- BF16、torch.compile、drop_last 及原验证/测试流程保持；MerRec 稀疏优化器分块设置保持。
- SISA lambda_init=0.1、score_scale=1.0、decay_bias=-5.0、exp_clamp=11.0。
- 四个最终设置均关闭外层激活检查点，使用原来的整块编译策略及默认编译器内存策略。

{coverage}

8、32 维与 16 维 seed 20262028、20262029 的覆盖规模一致；
64 维与 16 维 seed 20262027 的四卡覆盖规模一致。三次均值不对应一个单独测试样本集合。
更小的单卡 batch 减少了每个 parquet 块尾部被 drop_last 丢弃的样本数，
因此相同全局 batch 并不意味着相同的实际训练、验证和测试保留集合。
两卡各 rank 的训练批次数为 4,395、4,397；四卡为 4,405、4,395、4,392、4,395。
DDP 保留原有不等长输入处理；不能将全局 batch 恒定解释为最后几个同步步的有效样本数、
全部梯度聚合顺序或优化步数均完全相同。每个 rank 的完整覆盖记录随配置归档。
历史未保存逐样本身份、完整数据内容和完整软件环境的逐字节快照，
因此不声称历史与当前数据、源码及软件环境完全相同。

## 任务记录、OOM 和重试

{attempts}

16 维由表中三个历史完整 SISA 任务组成，六份原始日志已重新读取并与归档核对；
548166_1 仍是原始启动协议参考，但报告中的 16 维指标取三次等权均值。
新增三种维度使用独立 run ID，从头训练。64 维 attempt 1 在首个编译训练前向中失败：
两张 H100 80GB 的单卡显存均接近上限，额外申请 816 MiB 时仅剩约 533 MiB，
未完成一个训练批次，没有有效 Test 指标。该失败不能解释为 64 维的预测性能退化。

64 维 attempt 2 只增加 gradient_checkpointing=True，使用隔离配置，仍在首个前向中 OOM。
CPU FP32/BF16 的检查点输出和梯度等价检查通过，但这并不保证编译后的峰值显存下降。
后续四个短诊断任务保留了完整记录：643933 完成 GPU 等价检查后遇到诊断导入错误；
643996 的整块编译预算 0.0 通过前向但在反向 OOM；644025 的分层编译未通过 BF16 接近性检查；
644051 的整块编译预算 0.5 通过两个 rank 的 138 张量等价检查及三个原 batch 训练步，
小批量比较的最大绝对误差为 0，短诊断的峰值已分配显存约 76.76 GiB。
这些诊断没有最终 Test 指标，没有被当作正式效果结果，也没有把权重转给最终运行。

最终方案增加 GPU 数、降低单卡 batch、保持全局 batch。attempt 3 恢复原模型、
原整块编译和默认编译器设置，以四张 H100 从头完成训练；未采用上述诊断中的编译器内存改动。
本报告不把不同 GPU 数的耗时直接解释为纯维度速度变化，也不把 OOM 信息当作标准化显存基准。

守护按每轮 sleep 600 的方式检查进度并保留快照。期间修复了 Slurm 同时返回
具体运行记录与旧 PENDING 数组占位记录造成的状态覆盖问题，训练未受影响。
首轮失败、重试回执、源码快照、最终清单和完成状态均独立保留。

## 验证结论与解释边界

组成四个展示点的六个实际运行均为 COMPLETED/0:0，运行清单、有效配置、参数量、数据覆盖、
最终五标签指标及双日志一致性均通过检查。报告生成器另行从原始 Test 段解析指标，
用 Decimal 独立重算均值，共核对 60 个原始标签级指标值及 40 个展示值，避免仅信任汇总表。
初次提交前 141 项 CPU 测试通过，检查点重试前 145 项通过，最终四卡重试前 151 项通过。
修复并发进度条日志读取后，最终完整 CPU 回归共 152 项通过。

可以在明确限制下分享：只有一个模型—数据集组合；16 维为三次均值，其他维度各一次；
三次参考及不同维度之间存在 DDP 分片、内部种子和实际样本覆盖差异；跨维度的 SISA 初始化不保证数值嵌套；
参数容量和初始化尺度随维度变化。结果是固定预算下的描述性维度敏感性证据，
不构成统计显著性、因果机制分离或全部 16 个主实验组合的普遍结论。

本轮结论可表述为：使用 16 维三次均值作参考时，16 维平均 AUC 最高，64 维单次 LogLoss 最低；
64 维 AUC 比该均值略低，且只在 Offer 上高于参考。重复次数和协议差异限制了结论。
若要据此改变默认维度或用于更严格的单变量解释，应先统一重复次数、DDP 布局和保留样本进行复核；
若要声称最优维度稳定，还需要额外种子和组合。这些后续验证尚未运行。

## 归档与复现

- `report.md`：本独立报告；全部表值可从下面的证据重建。
- `archive/reports/generated_report.md`：生成器输出，独立于可手工修订的 report.md。
- `results/summary.csv`：正式展示表，16 维为三次均值，其他维度为单次，含重复次数和来源任务。
- `results/runs.csv`、`run_label_metrics.csv`：六个实际运行及 30 条逐运行标签记录，原始单次值完整保留。
- `results/label_metrics.csv`、`comparisons.csv`：20 条展示标签记录及相对三次均值的差值。
- `results/job_index.csv`：全部 12 个相关作业，明确区分正式运行、三个历史参考和诊断。
- `results/evidence.json`：全部接纳条件、来源散列、会计记录与失败尝试历史。
- `notes/final_validation.json`：报告的独立计算复核及解释边界。
- `notes/runtime_artifact_inventory.csv`：{len(artifacts)} 个运行产物的大小与 SHA-256，包括 {model_files} 个模型文件。
- `archive/pre_three_run_reference/`：修正前的单次参考报告、结果和审计记录，仅作历史追溯。
- `configs/`：原配置、历史对照、覆盖、任务映射和隔离的 64 维重试配置。
- `notes/submission_attempt1.json`、`submission_r64_attempt2.json`、`submission_r64_attempt3.json`、`submission.json`：原始与当前选择回执。
- `notes/r64_attempt1_failure.json`、`r64_attempt2_failure.json`、`attempt_history.json`、`r64_attempt3_protocol.csv`：失败与重试差异。
- `notes/source_sha256.json`、`source_sha256_r64_attempt2.json`、`source_sha256_r64_attempt3.json`：按尝试分别冻结的源码和配置身份。
- `notes/memory_probe_*.json`：短诊断任务、代码修复、数值结果及可行性检查；不纳入正式指标表。
- `notes/r64_four_gpu_coverage_comparison.json`：两卡与四卡的样本保留差异。
- `notes/launch.md`、`monitor_attempt1.json`、`monitor_attempt2.json`、`monitor.json`：启动、修复与守护记录。
- `artifacts/sisa_score_dim/`：运行清单、退出结果、原源码快照、全部守护快照及验证日志。
- `logs/` 和 `checkpoints/MerRec_Action/`：按 results/runs.csv 中的精确相对路径定位原始输出和检查点。

重建命令依次为 `.venv/bin/python scripts/collect_sisa_score_dim_results.py` 和
`.venv/bin/python scripts/build_sisa_score_dim_report.py`。生成器要求四个维度全部通过审计，
不会用未完成、失败或验证集指标代替最终结果，也不会默认覆盖可手工修订的 report.md。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DIRECTORY / "results/evidence.json")
    parser.add_argument("--output", type=Path, default=DIRECTORY / "archive/reports/generated_report.md")
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    cells, validation = verify(evidence)
    report = report_text(evidence, cells)
    if re.search(r"https?://|\]\(", report):
        raise ValueError("Standalone report must not contain hyperlinks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    validation["report_file"] = str(args.output.relative_to(ROOT))
    validation["report_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    validation["evidence_sha256"] = hashlib.sha256(args.evidence.read_bytes()).hexdigest()
    (DIRECTORY / "notes/final_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(output=str(args.output.relative_to(ROOT)),
                          validation=validation["status"], label_metric_values=40), ensure_ascii=False))


if __name__ == "__main__":
    main()
