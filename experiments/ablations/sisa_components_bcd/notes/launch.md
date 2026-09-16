# 提交与验证记录

日期：2026-09-16。

最终范围：RankMixer–KuaiRand，一个组合，B/C/D 三个任务，一个 seed。
数组 ID 0、1、2 分别对应 B、C、D。完整 SISA E 复用 H100 历史任务 548166_7。
用户已明确选择严格匹配已有 H100 轮次：seed 20262028、两卡、每卡 batch 16384，
数据加载 seed 2027、SISA 参数 seed 20260822。原始四卡 baseline 不作配对对照。
初始准备中的 HyFormer–MerRec 已按用户要求从配置、任务表和提交数组中移除；未提交过该组合。

提交前验证：

- 最终两卡协议全量 CPU 回归：96 项通过，包含新增 8 项组件和协议检查。
- 新增测试先验证旧实现不支持组件消融，再验证实现后的公式、梯度、配置传递和固定任务范围。
- 最终有效配置已与 H100 原始日志全部字段比对，三组共 219 项检查；
  差异仅为组件模式和输出 run ID，详见 configs/protocol_comparison.csv。
- 使用当前真实数据和 BlockedParquetBatchDataset 重建 train/valid/test 分配，
  两个 rank 的块数、行数、批次数均与历史 548166_7 完全一致，详见 configs/data_coverage.csv。
  该检查验证分配与覆盖，不等同于历史文件逐字节哈希校验。
- 三个最终任务 dry-run 均通过，固定两卡与 H100 复跑种子。
- run_expid.py --help 正常展示 --sisa-variant。
- 全部 Slurm 脚本通过 bash -n。
- 完整 SISA 的参数初始化及两种衰减参考下的 FP32 CPU 输出，与修改前 HEAD 逐元素完全一致。
- 尚无新增 GPU 训练结果；CPU 结果不作为正式实验结果。

原始验证输出保存在 artifacts/sisa_components_bcd/validation/，由 Git 忽略。
待提交成功后记录 job ID 和实际排队状态。
