# OneTrans SISA 三种子补齐：提交记录

提交时间：2026-09-18 16:16:47（Asia/Shanghai）。
提交实现：80f98401fc6a074b7d2cd4b56ce783201277fa2f。
数组作业：630948，数组范围0–9，最多四个任务并行。

用户明确选择仅补SISA：五个数据集各补seed20262028/20262029，共十个新任务。
旧seed20262027及全部baseline保持原记录，不另行提交。

| job/task | 数据集 | seed | 提交后核验状态 |
|---|---|---:|---|
| 630948_0 | QK-Video | 20262028 | PENDING / Priority |
| 630948_1 | QK-Video | 20262029 | PENDING / Priority |
| 630948_2 | KuaiRand | 20262028 | PENDING / Priority |
| 630948_3 | KuaiRand | 20262029 | PENDING / Priority |
| 630948_4 | TAAC-25 | 20262028 | PENDING / Priority |
| 630948_5 | TAAC-25 | 20262029 | PENDING / Priority |
| 630948_6 | Taobao | 20262028 | PENDING / Priority |
| 630948_7 | Taobao | 20262029 | PENDING / Priority |
| 630948_8 | MerRec | 20262028 | PENDING / Priority |
| 630948_9 | MerRec | 20262029 | PENDING / Priority |

上述为2026-09-18约16:17的即时状态，不表示训练已完成。
每任务4GPU、32CPU、960GiB主机内存、12小时；运行时检查GPU为H100。
原始Slurm输出使用logs/unirank-onetrans-seeds-630948_<task_id>.out。

启动前十个dry-run全部通过，逐项核验734个有效参数；唯一配置变化为run ID和
实验种子及其两个子流。四rank实际数据覆盖已预检，详见reference_audit.md。
TAAC-25通过study专用配置恢复历史group_id=user_id，主配置未修改。
启动器的任务映射、防覆盖、协议漂移及四卡预算测试先写后实现；提交前完整CPU
回归112项通过，新Slurm脚本语法及调度器test-only检查通过。

本次提交时仓库已有用户对消融report.md的未提交编辑，已原样保留且没有纳入提交。
新任务源码、配置及提交回执独立归档，原始数据、日志、模型和验证输出仍由Git忽略。

## 用户取消Taobao后的更新

2026-09-18 16:21:45，根据用户新指令仅取消630948_6和630948_7，对应Taobao的
seed20262028/20262029。取消时两任务尚未启动，不删除历史Taobao实验记录。
其余0、1、2、3、4、5、8、9共八个任务继续排队，原任务ID不重排。
原提交回执保留不变，取消命令与返回码另存cancellation.json。
启动器新增用户取消任务的拒绝检查；后续默认Slurm数组范围改为0-5,8-9%4，
避免重新提交已取消的Taobao。上方十任务表仍是最初提交时点的历史快照。

后续采集器与取消防护加入后，完整CPU回归122项全部通过，Slurm语法检查通过。
2026-09-18 16:24:07的results快照包含10个已核验历史运行、8个排队新任务和
2个用户取消记录，所有新任务指标均为空，尚无完整三seed汇总。
未启动即取消的两个数组元素已从squeue及sacct列表移除；采集器以成功取消回执
记录为CANCELLED_BY_USER，明确区别于调度器返回的完成状态，不伪造sacct取消行。
