# SISA 结果选择页

打开 `report.html`，点击模型×数据集单元格。默认在相同实验 seed 的有效运行中取 AUC 较高者，再选 AUC 最高的三个不同 seed 参与计算；页面标明有复跑可选和已选复跑。Baseline 每格固定一条。HiFormer 四数据集默认使用 target_only 新接入方式，旧完整接入以整组方式可切换，两种方式不会混算。

选择可手动改动并即时更新最终汇总表；可下载独立 Markdown 报告、CSV 或选择 JSON。按测试 AUC 选择后的均值是描述性展示，不代表预先固定的三种子估计。

在仓库根目录运行 `node experiments/sisa_three_seed_unified/report/deliver_report.mjs` 可重建页面。数据和来源散列见 `selector_catalog.json`；页面核验见 `selector_qa.json`。先前完整报告位于 `archive/20260924_before_selector/`。
