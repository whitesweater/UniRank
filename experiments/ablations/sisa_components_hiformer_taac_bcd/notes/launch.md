# 提交与验证记录

日期：2026-09-16。

新增范围：HiFormer–TAAC-25，一个组合、一个 seed，仅 B/C/D 三个任务。
前两组 RankMixer–KuaiRand（626644）和 HyFormer–MerRec（626760）保留，不重复提交。
完整 SISA 对照固定为 H100 历史任务 548166_14，seed 20262028。

提交前检查：

- 全量 CPU 回归 99 项通过；新增任务映射测试先确认旧映射拒绝第三组，再验证支持后的行为。
- 三个任务 dry-run 均通过；最终配置与原始 72 字段日志共进行 219 项比较。
  差异仅为 sisa_variant 与用于区分输出的 model_id。
- 当前数据两 rank × train/valid/test 的块数、行数和批次数全部匹配历史记录，
  详见 configs/data_coverage.csv；该检查不等同于完整历史数据内容散列核验。
- 新 Slurm 脚本通过 bash -n，数组固定 0–2，每任务 2 GPU、16 CPU、480 GiB 内存。
- 原两组映射保持兼容；第三组不携带 MerRec 专用优化器覆盖或 allocator 设置。
- 此次只新增 study 映射、配置、启动脚本和记录，未修改模型或训练实现。

原始验证输出保存于 artifacts/sisa_components_hiformer_taac_bcd/validation/，由 Git 忽略。
已提交数组作业 626779，提交时实现版本为 43c65c0。

| job/task | 组别 | 组件模式 | 提交后状态 |
|---|---|---|---|
| 626779_0 | B | content_only | PENDING / Priority |
| 626779_1 | C | decay_only | PENDING / Priority |
| 626779_2 | D | rotation_only | PENDING / Priority |

squeue 已核对三个元素均申请 acd_u、2 GPU、16 CPU、480 GiB 内存。
本次仅新增以上三个任务，前两组不重复提交。
原始 Slurm 日志命名为 logs/unirank-sisa-hiformer-taac-bcd-626779_<task_id>.out。
当前尚无第三组训练指标，提交成功与 CPU 检查通过不代表正式实验完成。
