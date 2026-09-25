# 低值种子复跑批次

本批次仅重跑 8 个已经完成、但指标明显偏低的 SISA 任务，各复跑一次。实验种子及其内部随机流不变，沿用各任务原训练参数。

筛选规则：前次宏 AUC 极差至少 0.005 的组合中，最低点距次低点至少 0.003；另补标签 AUC 极差至少 0.01 且最低点距次低点至少 0.008 的任务。去重后 7 个 AUC 低值任务；加上 OneTrans–TAAC-25 的 seed29 LogLoss 高值任务，共 8 个。

RankMixer 的突出 AUC 差距主要来自某一个种子较高，其余两个种子接近，因此本批次未选入。

| 模型 | 数据集 | 种子 | 核查重点 | 作业 |
|---|---|---|---|---|
| Zenith | TencentGR_10M_Action | 20262027 | macro_auc | 645223_12 |
| HyFormer | TencentGR_10M_Action | 20262027 | macro_auc | 645223_13 |
| HyFormer | MerRec_Action | 20262028 | macro_auc; Offer | 645224_1 |
| Zenith | MerRec_Action | 20262029 | macro_auc; Purchase | 645225_0 |
| HiFormer | MerRec_Action | 20262029 | Checkout | 645226_2 |
| HyFormer | KuaiRand_Video_Action | 20262029 | is_follow | 645226_5 |
| OneTrans | MerRec_Action | 20262028 | macro_auc | 645227_8 |
| OneTrans | TencentGR_10M_Action | 20262029 | macro_logloss | 645228_5 |

分支 main，提交 22515f5fd4fa35094e565cb23f6988d2b5cff87b。所有有效配置与原日志逐项相符，历史 seed27 未打印的两个内部随机流按当时默认值显式设置。当前 full-SISA 路径相对于三种子归档后的代码只增加消融可选分支，默认仍为 full。OneTrans 源码/配置散列与原 manifest 全部一致。

21 项注意力、SISA 分支、种子覆盖和启动器测试通过；既有 shell 启动器语法检查通过；OneTrans 两种布局 dry-run 通过。提交前及开始训练前均验证冻结源码/配置散列。

复跑使用新的 attempt 和运行 ID，原日志及原三种子统计保持保留。最终测试结果将单独写入本目录 results/ 与 report.md。仅作复跑对照，不自动按更高测试指标替换旧值。

后台本地进程每 60 秒检查队列、训练推进及最终结果，登记在 watcher.json。运行日志放在 artifacts/sisa_seed_variability_20260923/retry_watch/。原计划的 CPU-only Slurm 汇总任务被集群规则拒绝，未创建作业，现由本地检查进程完成汇总。总计仍为 8 个训练任务。
