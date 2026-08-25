# 实验档案

本目录是项目中实验计划、最终报告和可复现结果表的唯一入口。运行时大文件仍保存在
仓库根目录的 `logs/`、`checkpoints/` 和 `artifacts/`；这里只归档适合长期审阅与版本
控制的文档、配置快照、汇总表和图表。

## 实验索引

| 实验 | 状态 | 规模 | 入口 |
| --- | --- | ---: | --- |
| SISA 原生严格矩阵 | 已完成 | 32 tasks / 16 pairs | [报告](sisa_native_strict/report.md) · [结果](sisa_native_strict/results/) · [迁移报告](../reports/unirank_strict_migration_20260825/report.html) |
| SISA HPC3/ACD 扩展矩阵 | 已完成 | 38 tasks / 19 pairs | [报告](sisa_expansion_acd/report.md) · [结果](sisa_expansion_acd/results/) · [原计划](sisa_expansion_acd/planning.md) |
| 后续消融实验 | 待开展 | 按 study 建档 | [消融索引与规范](ablations/README.md) |

## 统一目录约定

每个正式实验使用一个稳定的 `study_slug`：

```text
experiments/<study_slug>/
├── report.md          # 结论、协议、异常恢复、局限性
├── planning.md        # 实验矩阵与完成清单（如适用）
├── configs/           # 本次实验专用配置或配置快照（如适用）
├── results/           # runs/metrics/paired summary 等小型机器可读结果
├── figures/           # 从 results 生成的图表
└── notes/             # 非最终分析记录；不得替代 report.md
```

目录中不放 Parquet、checkpoint、完整 Slurm stdout 或编译缓存。`report.md` 必须记录原始
日志位置、最终 job/task 映射和结果生成命令，使运行时证据仍可追溯。

新增消融实验时，从 [消融模板](templates/ablation/README.md) 开始，并同步更新本页索引。
