# 提交与验证记录

日期：2026-09-16。

新增范围：HyFormer–MerRec，一个组合、一个 seed，仅 B/C/D 三个任务。
第一组 RankMixer–KuaiRand 的作业 626644 保留，不重复提交。
完整 SISA 对照固定为 H100 历史任务 548166_1，seed 20262028。

提交前检查：

- 全量 CPU 回归 98 项通过，包含两组独立任务映射、历史参数比对及 MerRec 优化器和分配器设置检查。
- 新增测试先确认原启动器不能选择第二组，再确认扩展后的行为。
- 三个任务 dry-run 均通过；最终配置与原始 74 字段日志共进行 225 项比较。
  差异仅为 sisa_variant 与用于区分输出的 model_id。
- 两 rank × train/valid/test 的块数、行数和批次数全部匹配历史记录；详见 configs/data_coverage.csv。
- 新 Slurm 提交脚本通过 bash -n，数组固定 0–2，每任务 2 GPU、16 CPU、480 GiB 内存。
- 保留 sparse_optimizer_foreach=False、sparse_adagrad_chunk_size=16777216 和 expandable_segments:True。
- 原组的默认启动参数、任务 ID 和配置快照保持兼容；此次未修改模型或训练实现。

原始验证输出归档在 artifacts/sisa_components_hyformer_merrec_bcd/validation/，由 Git 忽略。
当前记录只表示提交准备完成，尚无新组训练结果。提交后补记实际 job ID 和调度状态。
