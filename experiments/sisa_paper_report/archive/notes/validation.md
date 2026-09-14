# 论文实验报告验证说明

审计日期：2026-09-14（Asia/Shanghai）。结论：**归档数值及汇总可分享，但必须附带实验局限；首次代码检查未通过，后续已修复并通过完整 CPU 回归；未重新训练。**

## 已完成的数值与协议检查

- 正式范围：4 个模型 × 4 个数据集，16 个单元、68 个模型–数据集–标签组合。
- 64 次最终运行均有原始 Slurm 日志与 checkpoint 指标日志；272 组最终测试 AUC/Logloss（544 个标量）与 runs.csv 的 JSON 和 metrics.csv 精确一致。
- 32 次旧协议运行构成 16 个 baseline/SISA 配对，各配对 GPU 型号一致；所选训练字段除 SISA 设置和 Zenith–MerRec 的优化器分块兼容设置外无差异。
- 旧配对和后两轮 SISA 配对在各 rank 的 train/valid/test 分配行数、处理行数、步数均一致。旧协议与新协议之间的 drop_last 差异被明确保留。
- 152 个 Parquet 文件的 footer 行数，与元信息及日志中全部 12 个划分的样本数一致；未读取完整样本内容，也未重算原始预测的 ROC-AUC。
- 独立重算三轮单元 AUC 均值、样本 SD 与 Logloss 差值，与原统一汇总误差不超过 1e−12。
- 重复会话处理：543255_18 的两段会话采用最后测试段，不增加 seed 数；549797_0 attempt2 是 seed20262029 task0 的正式结果。
- 主配对 ΔAUC 为 +0.005840961458333327；三轮描述性 ΔAUC 为 +0.006414925694444437，均采用 16 单元宏平均。

## 测试与工作区限制

```bash
.venv/bin/python -m unittest discover -s experiments/sisa_paper_report/archive/tests -p test_audit_sisa_paper_evidence.py -v
.venv/bin/python -m py_compile experiments/sisa_paper_report/archive/scripts/audit_sisa_paper_evidence.py experiments/sisa_paper_report/archive/scripts/build_sisa_paper_report.py
```

新增审计测试 **5/5 通过**，覆盖范围过滤、去重与缺失标签、最终测试边界、多会话最终结果选择和不等标签数量下的单元宏平均。两个新脚本编译通过。

全量 CPU suite 本次运行 **70 项，3 个导入错误**。错误来自工作开始前已有的 `unirank/pytorch/layers/attentions/sisa.py:20` 语法错误 `channel_dim: int,+2`，影响 `test_extended_sisa_adapters`、`test_sisa_attention`、`test_sisa_single_seed_acd`。该次全量运行早于新增审计测试写入；新增的 5 项已单独运行。没有修改或移除用户的源码改动，也没有把历史通过记录当作当前通过。

全仓 `git diff --check` 还指出先前已修改的 `experiments/sisa_native_strict/migration/report.html:32` 存在尾随空白。本次未修改该历史 HTML；新改动单独检查。

未修改 Slurm 脚本；未启动训练、重新评测、提交任务、提交 Git commit 或发布外部网站。

## 必须随论文保留的限制

1. Baseline 只有 seed20262027；后两轮 SISA 不能合成三轮配对实验。
2. World size 和 per-GPU batch 改变了尾批训练/评测样本集合；全局 batch 相同不能消除混杂。
3. 68 个标签组合并非独立随机重复；不提供统计显著性或从混合协议三点 SD 推导 CI。
4. MerRec 的 metadata 块布局过时：32/16/16，对应当前文件与运行日志为 8/8/8；总行数一致。
5. 元信息描述的时间划分、正样本约束和标签构造未经逐事件泄漏审计；未新增正例率、历史长度分布或真实唯一用户数统计。
6. 论文四位小数转录只作次级复现参考，原文版本与表号需投稿前再核验。
7. 无完成的正式 SISA 性能消融、配对效率测量或线上实验；这些是待补证据。

## 报告构建与阅读检查

按用户最终要求，只交付一份独立中文报告 `report.md`。正文含 25 张完整表格，包括实验设置总表、逐单元硬件、全部 68 个标签组合的 baseline、三轮 AUC/Logloss、均值和 SD。所有数值均在正文，不依赖其他文档；不交付 HTML 或外链图表。

已检查正文：超链接为 0，外部文件阅读依赖为 0；训练与评测设置、正负结果和必要限制均保留。报告由 `experiments/sisa_paper_report/archive/scripts/build_sisa_paper_report.py` 生成。

## 后续归档与提交检查

按用户进一步要求清理 Git 工作区后，SISA 参数声明中的误输入已修复。完整 CPU suite
88/88 通过，归档审计测试 5/5 通过；此前 70 项中的 3 个导入错误是保留的历史记录。
报告生成脚本按归档路径重跑成功，移动操作本身未改变最终报告的任何字节；随后仅更新
报告中的代码检查状态，实验数字、实验设置和无超链接要求保持不变。

原始日志、失败截图保留在 archive/logs 与 archive/previews 并加入 Git 忽略规则。
历史 HTML 的空白、构建入口根目录和逐动作 SQL 来源已修正；canonical artifact 校验通过。
