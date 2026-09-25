# SISA 结果选择页

打开 `report.html`，点击模型×数据集单元格，选择三个不同的 SISA seed；同种子复跑可二选一。Baseline 每格仅有一次，固定显示。选择即时更新最终汇总表，可以下载独立 Markdown 报告、CSV 或选择 JSON。

在仓库根目录运行 `node experiments/sisa_three_seed_unified/report/deliver_report.mjs` 可重建页面。数据和来源散列见 `selector_catalog.json`；页面核验见 `selector_qa.json`。先前完整报告位于 `archive/20260924_before_selector/`。
