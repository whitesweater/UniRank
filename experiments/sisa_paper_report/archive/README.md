# 报告整理过程归档

归档日期：2026-09-14。

此目录集中保存本次论文实验报告整理产生的中间材料。最终独立报告仍位于
`experiments/sisa_paper_report/report.md`，正文不依赖本目录阅读。

## 目录说明

| 目录或文件 | 内容 |
| --- | --- |
| `results/` | 独立复算结果、配置快照、覆盖率审计、数据元信息和来源清单 |
| `scripts/` | 本次使用的证据审计脚本和报告生成脚本 |
| `tests/` | 证据范围、指标选择与汇总计算的 5 项回归测试 |
| `notes/` | 来源说明与验证记录 |
| `logs/` | 原始 CPU 检查输出、审计摘要、未交付 HTML 的构建失败记录 |
| `previews/` | 未交付 HTML 的验证失败截图，仅保留过程记录 |
| `archive_manifest.json` | 文件原位置、归档位置、归档前后 SHA-256 及最终报告校验值 |

仅移动本次任务创建的文件；历史实验、原始训练日志、模型源码及工作开始前已有的
改动均保留。HTML 构建产物未作为最终报告交付，失败记录不代表论文数值审计失败。
可再生成的 Python 字节码缓存不归档。

## 在仓库根目录使用

仅重新生成独立报告，不重新审计原始训练日志：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/sisa_paper_report/archive/scripts/build_sisa_paper_report.py
```

运行归档内的审计测试：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s experiments/sisa_paper_report/archive/tests -v
```

若需要重新读取正式训练日志及数据 footer，执行下列命令。它会更新 `archive/results/`
中的审计快照；相关运行时文件必须存在，且不涉及提交训练任务。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/sisa_paper_report/archive/scripts/audit_sisa_paper_evidence.py
```

归档脚本已按新位置修正路径。原始日志内容保持原样；说明文档中的运行命令已更新。
归档测试不再由顶层 `tests/` 自动发现，须使用上面的专用测试命令。

归档后修复了此前阻碍模型导入的语法错误，完整 CPU 回归 88 项及归档审计测试 5 项均通过。
最终报告仅同步更新这项验证状态，所有实验设置与结果表保持不变。日志与失败截图
保存在本地归档并由 Git 忽略；其余紧凑材料纳入版本控制。
