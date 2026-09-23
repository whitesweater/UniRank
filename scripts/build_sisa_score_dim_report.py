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
    if not evidence.get("complete") or len(rows) != 4:
        raise ValueError("Final report requires four audited dimensions")
    cells = {row["score_dim"]: row for row in rows}
    if set(cells) != set(DIMENSIONS):
        raise ValueError("Dimension coverage differs from the authorized study")
    checks = []
    for dim, row in cells.items():
        if not row["eligible"] or row["state"] != "COMPLETED" or row["exit_code"] != "0:0" or row["issues"]:
            raise ValueError(f"Unaccepted evidence at r={dim}")
        parsed = independent_metrics(ROOT / row["checkpoint_log"])
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
        coverage_file = "data_coverage_ws4.csv" if dim == 64 else "data_coverage.csv"
        with (DIRECTORY / "configs" / coverage_file).open() as stream:
            expected_coverage = [{key: value if key == "split" else int(value)
                                  for key, value in record.items() if key != "status"}
                                 for record in csv.DictReader(stream)]
        if row["coverage"] != expected_coverage:
            raise ValueError(f"Per-rank sample coverage differs from its frozen layout at r={dim}")
        world_size, per_gpu_batch = (4, 8192) if dim == 64 else (2, 16384)
        if (row.get("world_size"), row.get("per_gpu_batch"), row.get("global_batch")) != (world_size, per_gpu_batch, 32768):
            raise ValueError(f"Incorrect global-batch layout: {dim}")
        if row.get("gradient_checkpointing") is not False:
            raise ValueError(f"Execution adjustment not explicitly disclosed: {dim}")
        scope = "dimension_with_four_gpu_layout" if dim == 64 else "dimension_only"
        if row.get("comparison_scope") != scope:
            raise ValueError(f"Wrong comparison scope: {dim}")
        expected_total = cells[16]["total_parameters"] + 7680 * (dim - 16)
        expected_dense = cells[16]["dense_parameters"] + 7680 * (dim - 16)
        if row["total_parameters"] != expected_total or row["dense_parameters"] != expected_dense:
            raise ValueError(f"Parameter count mismatch: {dim}")
        for field in ("checkpoint_log", "slurm_log"):
            path = ROOT / row[field]
            checks.append(dict(score_dim=dim, path=row[field],
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    history = evidence.get("attempt_history", [])
    if (len(history) != 2 or {row["job_task_id"] for row in history} != {"643736_2", "643901_2"}
            or any(row["eligible"] for row in history)):
        raise ValueError("Both failed r64 attempts must remain separately visible")
    validation = dict(status="share_with_caveats", dimensions=4, labels_per_dimension=5,
                      independently_verified_label_metric_values=40,
                      macro_calculation="Exact Decimal mean of five final Test labels",
                      comparison_groups={"dimension_only": [8, 16, 32],
                                         "dimension_with_four_gpu_layout": [64]},
                      issues=[], caveats=["one seed and one model-dataset pair",
                          "r64 uses four GPUs at batch 8192, other dimensions two GPUs at batch 16384",
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
    summary_rows = []
    for dim in DIMENSIONS:
        row = cells[dim]
        summary_rows.append([str(dim), f'{row["macro_auc"]:.7f}', f'{row["macro_logloss"]:.7f}',
                             f'{row["macro_auc"] - reference["macro_auc"]:+.7f}',
                             f'{row["macro_logloss"] - reference["macro_logloss"]:+.7f}',
                             f'{row["world_size"]}×{row["per_gpu_batch"]:,}'])
    summary = table(["每头维度", "宏平均 AUC ↑", "宏平均 LogLoss ↓", "ΔAUC", "ΔLogLoss", "GPU 数×单卡 batch"], summary_rows)
    auc = table(["任务", "8 维", "16 维", "32 维", "64 维（四卡）"],
                [[label, *[f'{cells[dim]["metrics"][label]["AUC"]:.6f}' for dim in DIMENSIONS]] for label in LABELS])
    loss = table(["任务", "8 维", "16 维", "32 维", "64 维（四卡）"],
                 [[label, *[f'{cells[dim]["metrics"][label]["logloss"]:.6f}' for dim in DIMENSIONS]] for label in LABELS])
    label_changes = []
    for dim in (8, 32, 64):
        auc_wins = sum(cells[dim]["metrics"][label]["AUC"] > reference["metrics"][label]["AUC"] for label in LABELS)
        loss_wins = sum(cells[dim]["metrics"][label]["logloss"] < reference["metrics"][label]["logloss"] for label in LABELS)
        label_changes.append(f"- {dim} 维相对 16 维：{auc_wins}/5 个标签 AUC 更高，{loss_wins}/5 个标签 LogLoss 更低。")
    params = table(["每头维度", "旋转平面/头", "SISA 分支参数", "全模型可训练参数", "稠密可训练参数"],
                   [[str(dim), str(dim // 2), f"{7680 * dim + 3096:,}",
                     f'{cells[dim]["total_parameters"]:,}', f'{cells[dim]["dense_parameters"]:,}'] for dim in DIMENSIONS])
    coverage_rows = []
    for split in ("train", "valid", "test"):
        before = sum(row["processed_rows"] for row in reference["coverage"] if row["split"] == split)
        after = sum(row["processed_rows"] for row in cells[64]["coverage"] if row["split"] == split)
        coverage_rows.append([split, f"{before:,}", f"{after:,}", f"{after-before:+,}",
                              f"{(after-before)/before*100:+.4f}%"])
    coverage = table(["划分", "两卡协议保留行数", "四卡协议保留行数", "差值", "相对变化"], coverage_rows)
    attempts = table(["维度", "任务", "尝试", "状态", "处理"],
                     [[str(dim), cells[dim]["job_task_id"], str(cells[dim]["attempt"]), "COMPLETED / 0:0",
                       "两卡原设置" if dim != 64 else "四卡，单卡 batch 减半"] for dim in DIMENSIONS]
                     + [["64", "643736_2", "1", "FAILED / 1:0", "两卡原设置首批 OOM"],
                        ["64", "643901_2", "2", "FAILED / 1:0", "两卡外层检查点仍在首批 OOM"]])
    fixed = max((8, 16, 32), key=lambda dim: cells[dim]["macro_auc"])
    best_auc = max(DIMENSIONS, key=lambda dim: cells[dim]["macro_auc"])
    best_loss = min(DIMENSIONS, key=lambda dim: cells[dim]["macro_logloss"])
    delta64 = cells[64]["macro_auc"] - reference["macro_auc"]
    return f"""# HyFormer–MerRec SISA 投影维度消融报告

## 结果摘要

本轮已获得每头 8、16、32、64 维的完整测试结果，五个行为标签均已核验。
在已记录训练参数和两卡布局对齐的 8、16、32 维中，{fixed} 维的宏平均 AUC 最高。
四个已观察设置中，宏平均 AUC 最高的是 {best_auc} 维，LogLoss 最低的是 {best_loss} 维。
64 维的 AUC 提升集中在 Offer，其余四个标签的 AUC 均略低于 16 维，不能表述为所有任务都改善。
64 维的两次两卡尝试因显存不足失败，最终采用四卡、每卡 batch 8,192，
全局 batch 仍为 32,768，使用原模型和编译设置完成重试。
其 AUC 相对 16 维的差值为 {delta64:+.7f}。四卡改变了 DDP 分片和 drop_last 实际覆盖，
64 维是带资源布局差异的描述性比较，不能将其差值完全归因于维度。

结果截至 {evidence['generated_at']}。本报告只有 HyFormer–MerRec、seed 20262028，
不支持统计显著性、跨种子稳定性或其他模型和数据集的普遍结论。

## 维度含义与评价口径

研究变量为 `sisa_score_dim`，即每个注意力头的 B、C 投影通道数。
完整 SISA 在这些通道上执行累计衰减和二维成对旋转，形成 softmax 之前的附加分数。
骨干 token_dim=256、num_heads=2，原有每头 Q/K 维度为 128；本轮保持头数不变。
单个 B 或 C 投影在两个头上的总宽度为 2r，每头有 r/2 个旋转平面。
三个序列注意力和三个 query-to-sequence 注意力接入点统一改变 r。

五个标签为 Like、Cart、Offer、Checkout、Purchase。宏平均为五个标签各自指标的
等权算术平均，不是把五个标签预测拼起来重新计算的 AUC，也不是按标签阳性数加权。
AUC 越高越好，LogLoss 越低越好；Δ 均为当前维度减去预指定的 16 维对照，单位为指标原值。
所有指标来自最终 Test evaluation，checkpoint 按原验证集宏平均 AUC 规则选择。
未根据测试结果删去退化设置或更换参考任务。
宏平均由日志中保留六位小数的标签指标重新计算，表内七位小数是算术展示精度，
不代表新增的测量精度或统计显著性。

## 宏平均结果与执行差异

{summary}

原设置下，8 维和 32 维的宏平均 AUC、LogLoss 均未超过 16 维。
这说明本组合在该种子和训练预算下没有呈现“维度越大，宏平均效果越好”的规律。
64 维行保留真实结果，同时标明四卡布局；其测试保留样本数较两卡协议增加 40,960 行。

## 各行为任务的变化并不一致

### AUC

{auc}

### LogLoss

{loss}

{chr(10).join(label_changes)}

32 维的 Offer AUC 从 0.731796 提高到 0.754788，但其他四个标签 AUC 下降，
因此 Offer 的局部提升没有转化为更高的宏平均 AUC。
64 维的 Offer AUC 相对 16 维提高 0.023456，对宏平均的贡献为 +0.0046912；
其余四个标签合计贡献 -0.0011620，净增为 +0.0035292。LogLoss 的宏平均改善
主要来自 Like，其下降 0.004659，对宏平均贡献 -0.0009318。
8 维的五个标签 AUC 均低于 16 维。宏平均与逐标签结果应同时报告，
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
- 8、16、32 维：两卡 DDP，每卡 batch 16,384；64 维：四卡 DDP，每卡 batch 8,192。
- 所有最终设置的名义全局 batch 均为 32,768，梯度累积均为 1。
- 训练 1 epoch，max_len=100，token_dim=256，num_heads=2，num_layers=3。
- 实验 seed=20262028，各 rank 使用 seed+rank；loader seed=2027；SISA 参数 seed=20260822。
- AdamW 稠密学习率 0.0001、weight_decay=0；Adagrad 稀疏学习率 0.05。
- BF16、torch.compile、drop_last 及原验证/测试流程保持；MerRec 稀疏优化器分块设置保持。
- SISA lambda_init=0.1、score_scale=1.0、decay_bias=-5.0、exp_clamp=11.0。
- 四个最终设置均关闭外层激活检查点，使用原来的整块编译策略及默认编译器内存策略。

{coverage}

8、16、32 维之间的覆盖逐项一致；64 维与预先重算的四卡覆盖逐项一致。
更小的单卡 batch 减少了每个 parquet 块尾部被 drop_last 丢弃的样本数，
因此相同全局 batch 并不意味着相同的实际训练、验证和测试保留集合。
两卡各 rank 的训练批次数为 4,395、4,397；四卡为 4,405、4,395、4,392、4,395。
DDP 保留原有不等长输入处理；不能将全局 batch 恒定解释为最后几个同步步的有效样本数、
全部梯度聚合顺序或优化步数均完全相同。每个 rank 的完整覆盖记录随配置归档。
历史未保存逐样本身份、完整数据内容和完整软件环境的逐字节快照，
因此不声称历史与当前数据、源码及软件环境完全相同。

## 任务记录、OOM 和重试

{attempts}

16 维来自预先指定的历史完整 SISA 任务 548166_1，已重新读取两份原始日志并与归档核对。
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

最终四个运行均为 COMPLETED/0:0，运行清单、有效配置、参数量、数据覆盖、
最终五标签指标及双日志一致性均通过检查。报告生成器另行从原始 Test 段解析指标，
用 Decimal 独立重算宏平均，共核对 40 个标签级指标值，避免仅信任汇总表。
初次提交前 141 项 CPU 测试通过，检查点重试前 145 项通过，最终四卡重试前 151 项通过。
修复并发进度条日志读取后，最终完整 CPU 回归共 152 项通过。

可以在明确限制下分享：本轮只有一个种子和一个模型—数据集组合；
64 维带有 DDP 分片和实际样本覆盖差异；跨维度的 SISA 初始化不保证数值嵌套；
参数容量和初始化尺度随维度变化。结果是固定预算下的描述性维度敏感性证据，
不构成统计显著性、因果机制分离或全部 16 个主实验组合的普遍结论。

本轮结论可表述为：64 维在已观察的四个设置中宏平均表现最好，而原两卡协议内 16 维最好；
64 维的 AUC 收益集中在 Offer，并伴随四卡布局和样本保留变化。
若要据此改变默认维度或用于更严格的单变量解释，应先统一 DDP 布局和保留样本进行复核；
若要声称最优维度稳定，还需要额外种子和组合。这些后续验证尚未运行。

## 归档与复现

- `report.md`：本独立报告；全部表值可从下面的证据重建。
- `archive/reports/generated_report.md`：生成器输出，独立于可手工修订的 report.md。
- `results/runs.csv`、`label_metrics.csv`、`comparisons.csv`：四个最终设置、20 个标签记录及相对差值。
- `results/job_index.csv`：全部 10 个相关作业，明确区分正式运行、历史参考和诊断。
- `results/evidence.json`：全部接纳条件、来源散列、会计记录与失败尝试历史。
- `notes/final_validation.json`：报告的独立计算复核及解释边界。
- `notes/runtime_artifact_inventory.csv`：31 个运行产物的大小与 SHA-256，包括四个模型文件。
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
