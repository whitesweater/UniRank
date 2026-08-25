# UniRank SISA 严格实验交接文档

## 实验范围与最终状态

最终实验矩阵包含 32 个逻辑任务：

- 模型：OneTrans、HiFormer、RankMixer、Zenith；
- 数据集：QK-Video、KuaiRand、Taobao、MerRec；
- 实验设置：上游 baseline 和加入 SISA attention bias 的变体。

**最终状态：32 个严格实验已全部完成。** 最后一个任务 `14854_27` 于
2026-08-22 23:38（CST）写入成功标记。2026-08-24 重新运行最终收集器得到
`strict_complete=32/32`；已经核验 task ID 0-31、全部 16 组同型号 GPU 的
baseline/SISA 配对、协议头、有限测试指标、所选日志错误数为 0，以及完整的成功标记。
当前 Slurm 队列为空，后台监控器在最后一次采样后正常退出。

此前的单卡数组 `14775` 和 `14798` 仅属于探索性 pilot。它们的 global batch 为
8192，不能与论文的四卡协议直接比较。2026-08-22 已取消其中所有尚未启动的数组
元素；已经完成或当时正在运行的元素仅作为诊断证据保留，不进入最终严格结果表。

严格实验使用的 Slurm 作业链如下：

| 用途 | Job | 资源 |
|---|---:|---|
| OneTrans-Taobao baseline 校准 | `14806_4` | 4 × RTX 4090 |
| 结构化校准审计 | `14809` | 仅 CPU 的门禁任务 |
| 主 RTX 矩阵 | `14810` | 已完成的 RTX 任务；未启动的 11/26/27 已取消 |
| L40S 矩阵 | `14811` | 4 × L40S，一次运行一个数组元素 |
| Zenith-MerRec baseline 重试 | `14843_30` | 4 × L40S |
| Zenith-MerRec SISA 重试 | `14846_31` | 4 × L40S |
| HiFormer-KuaiRand 配对重跑 | `14850_10`、`14850_11` | 4 × L40S，串行运行 |
| Zenith-KuaiRand 配对重跑 | `14854_26`、`14854_27` | 4 × L40S，串行运行 |

最终选择的校准、主矩阵和重试任务恰好覆盖 task ID 0-31 各一次。每一组偶数/奇数
task ID 对应的 baseline/SISA 都使用同一型号 GPU。

2026-08-22 15:53，另一位用户的两卡 RTX 交互作业已经阻塞所有四卡 RTX 任务超过
六小时。因此，尚未启动的 Zenith-KuaiRand 配对 `14810_26` 和 `14810_27` 最初被
原地改为申请四张 L40S。19:51 的开跑前审计发现，虽然 Slurm 表面显示 L40S 分区和
GRES，其底层 TRES 仍同时计入 4 张 RTX 4090 和 4 张 L40S。两个元素均在运行前取消
（`Elapsed=00:00:00`），随后以串行数组 `14854_[26-27%1]` 干净重提。新数组经审计
只申请一个节点、384 GiB 内存和 `gres/gpu:l40s=4`；配对两侧的 GPU 型号及全部训练
设置保持一致。

17:25，剩余两张 RTX 也被外部的一日交互任务占用，四卡 RTX 调度因此不可用。为避免
HiFormer-KuaiRand baseline 与 SISA 使用不同型号 GPU，任务 10 和 11 以配对 L40S
数组 `14850` 重跑，申请 384 GiB 节点内存且一次只运行一个元素。尚未启动的原 RTX
任务 `14810_11` 被取消。已完成的 RTX baseline `14810_10` 仅作为诊断证据保留；
最终矩阵为任务 10 和 11 统一选择 `14850` 的结果。

校准于 2026-08-22 成功完成。结构化门禁通过，严格 OneTrans-Taobao baseline 的测试
AUC 分别为 0.629803（click）、0.742172（cart）、0.780022（favor）和
0.767930（buy）；相对官方结果的差值分别为 +0.000272、-0.003624、-0.003258 和
+0.000644。门禁于 04:46 释放 `14810` 和 `14811`。

## 严格实验协议

每个严格任务的 Slurm 日志和 checkpoint 日志都必须具备以下证据：

- `world_size=4`，并使用四张同型号 GPU；
- 每卡 batch 8192，global batch 32768；
- accumulation steps 为 1，训练 1 epoch，seed 为 20262027；
- 最大序列长度为 100，启用 BF16；
- 使用 blocked DDP 训练，并对验证集和测试集进行分布式聚合；
- 保留 Slurm 分配的 `CUDA_VISIBLE_DEVICES`，不得覆盖；
- 测试指标均为有限值，并在最后写入 `SISA_NATIVE_STRICT_COMPLETE`。

`scripts/submit_sisa_native_strict.sbatch` 会在启动四个 `torchrun` worker 前检查静态
配置。校准任务 `14806_4` 后接 `scripts/gate_onetrans_taobao_calibration.sbatch`，后者
检查结构化协议证据，通过后才释放其余数组。

与官方 OneTrans-Taobao AUC 的比较属于诊断项，而不是要求数值完全相同的硬门禁。
当绝对差值超过 0.01 时，必须检查代码、数据、DDP 分片、优化器更新次数和 checkpoint
选择。上述项目确认正确后，硬件或软件版本造成的残余差异会被明确记录，但不会单独
阻止完整实验矩阵继续运行。

## 代码与数据审计

- 仓库 HEAD 与上游一致，为
  `90bb99a60724e975ec565836266fd86dcad0039d`。
- `config/model_config.yaml` 未修改。`config/dataset_config.yaml` 的改动仅用于将作者
  环境中的文件系统路径替换为本地数据路径。
- DDP 训练/评测、blocked 数据加载、checkpoint 选择和指标实现均保持上游行为。
  Zenith-MerRec 重试所需的定向优化器兼容改动在下文单独说明。
- 对仓库中 14 份可用的官方 benchmark 日志与本地合并配置进行了比较；batch size、
  accumulation、epoch、seed、最大序列长度和 embedding dimension 全部一致。缺少
  官方 base 日志的两组 RankMixer 实验仍使用未修改的上游配置。
- 关闭 SISA 时，任何原生 attention 位置都不会注册 SISA 参数。OneTrans、HiFormer、
  Zenith 和 target attention 的零 bias 对照与 baseline 输出精确一致；RankMixer 只在
  其已有的 target-attention pooling 位置接入 SISA，原 token mixer 未修改。
- SISA 的每头权重 `lambda_h` 是可学习参数：底层保存为 `nn.Parameter`，通过
  `softplus` 映射为正数。回归测试验证其梯度有限且非零，并确认一次 optimizer step
  后参数值确实发生变化。
- 严格 OneTrans-QK baseline 的四个标签均在官方测试 AUC 的 0.0014 范围内。
  OneTrans-MerRec 的 Like、Checkout 和 Purchase 在 0.01 内；Cart 为 -0.018662，
  Offer 为 +0.012236。后续审计确认 world size、batch/accumulation/epoch/seed/最大
  序列长度、特征定义、优化器参数量、总参数/稠密参数/embedding 参数量、rank 0 的
  训练/验证/测试样本数、4,405 次本地更新、DDP 模式和 checkpoint 选择路径全部
  一致。Baseline 未启用 SISA，exact-off 测试通过。未发现代码或数据不一致，因此这
  两个残余差值作为不同硬件/软件栈下的复现波动保留，不用于阻塞矩阵。
- 严格 OneTrans-KuaiRand baseline 的 click、like、comment 和 long-view 均在官方
  测试 AUC 的 0.009587 范围内；follow 为 +0.011582，forward 为 +0.010106。定向
  审计确认 world size、batch/accumulation/epoch/seed/最大序列长度、优化器类型和
  学习率、稠密/稀疏/总参数量、训练/验证/测试样本数、rank 0 的 7,814 次更新、BF16、
  blocked DDP 和最佳 checkpoint 测试路径全部一致。exact-off 回归测试同样通过，
  因此这两个临界差值作为环境级波动保留，不阻止 SISA 配对实验。
- 本地环境为 Python 3.12.12、PyTorch 2.8.0+cu126、CUDA runtime 12.6、
  cuDNN 9.10.2 和 PyArrow 25.0.1。论文使用 H20 GPU，本矩阵使用 RTX 4090 和
  L40S；上游依赖没有固定作者使用的精确 PyTorch/CUDA 版本。
- TAAC-25 于 2026-08-22 03:09 下载完成。Hugging Face dry-run 显示 103 个文件中
  缺失 0 个，PyArrow 成功打开全部 96 个 parquet 文件。当前 UniRank checkout 通过
  软链接指向唯一的一份实体数据目录，没有复制数据。

保留的探索性任务 `14775_2`（OneTrans-KuaiRand baseline）在验证进行到 93% 时被
Slurm CPU 内存 cgroup 终止。它已经完成训练，终止前没有 CUDA、数值或模型代码错误。
该 pilot 只申请了 64 GiB。为了在不改变任何训练超参数的情况下为各 rank 留出余量，
门禁释放数组前，将尚未启动的 8 个严格 KuaiRand 任务（`2`、`3`、`10`、`11`、
`18`、`19`、`26`、`27`）从 256 GiB 调整为 384 GiB 节点内存。节点总内存为
1,100,000 MiB，因此 RTX 和 L40S 严格任务仍可各运行一个。

Zenith-MerRec 的第一次尝试 `14810_30` 和 `14810_31` 在 RTX 4090 上首次参数更新前
失败。PyTorch 2.8 为 2,258,770,656 个稀疏参数选择了 CUDA Adagrad 的 `foreach`
实现；已分配约 42.6 GiB 后，第一次 step 还需额外申请 7.84 GiB multi-tensor 临时
空间，超过 48 GiB 显存。这是软件/硬件内存兼容问题：论文使用 H20，上游环境也没有
固定精确 PyTorch 版本。第一次重提 `14840` 从未启动；另一位用户的两卡、七日 RTX
作业使四卡 RTX 无法调度，所以该任务在 pending 状态下取消。重试数组 `14843` 为
baseline 和 SISA 统一使用四张 L40S，并且只显式设置稀疏 Adagrad
`foreach=False`。优化器、学习率、模型、每卡 batch、global batch、更新次数、seed、
精度和 epoch 均未改变；single-tensor 实现执行相同的 Adagrad 更新，但临时显存峰值
更低。其他任务仍使用自动默认路径。当时 27 个回归测试通过。失败尝试的证据集中保留
在 `logs/failed_attempts/20260822_zenith_merrec_cuda_oom/`，没有与最终日志混放。

11:53，`14843_30` 使用 `sparse_optimizer_foreach=False` 成功完成训练、验证和测试。
配对的 SISA 任务具有更高 activation 显存，并在完成一次更新后，因 PyTorch
single-tensor Adagrad 仍试图一次性生成完整的 7.84 GiB `sqrt(state_sum)` 临时张量而
再次 OOM。任务 `14846_31` 将完全相同的逐元素 Adagrad 公式按 16,777,216 个元素分块
计算，每块临时空间约 64 MiB，同时启用 expandable CUDA allocator。零容差回归测试
确认，连续三次分块更新及其 state sum 与标准
`torch.optim.Adagrad(foreach=False)` 精确一致。除该定向重试外，所有默认路径保持
不变；最终 29 个测试通过。第二次失败证据保存在
`logs/failed_attempts/20260822_zenith_merrec_cuda_oom/l40_foreach_false/`。

14:19，最终重试 `14846_31` 成功完成，耗时 1:15:25，ExitCode 为 `0:0`，使用四张
L40S、`world_size=4`，并写入严格完成标记。它无 OOM、无数值错误地运行完 rank 0 的
全部 4,405 次更新。测试 AUC 为 0.759388（Like）、0.814763（Cart）、
0.766747（Offer）、0.853138（Checkout）和 0.852104（Purchase）。相对配对 L40S
baseline `14843_30` 的 AUC 差值依次为 +0.014883、+0.007855、-0.006541、
+0.000843 和 -0.001119。该结果验证了定向显存兼容方案。14:49 巡检时严格矩阵达到
23/32，`14811_2` 已使用释放出的四卡 L40S 资源继续运行。

最终 Zenith-KuaiRand baseline/SISA 配对由 L40S 串行数组 `14854` 运行。Task 26 在
21:57 完成 rank 0 的 7,814 次更新；Task 27 在 23:38 完成相同更新次数。两者都使用
四张 L40S、384 GiB 节点内存和严格协议，所选日志没有错误匹配。Baseline 测试 AUC
为 0.781888（click）、0.881683（follow）、0.928141（like）、0.897885（comment）、
0.871088（forward）和 0.797290（long-view）；配对 SISA 差值依次为 +0.010871、
+0.008525、+0.002325、+0.003329、+0.002531 和 +0.012819。

## 最终实验结果

严格矩阵共有 76 个配对任务指标。SISA 使其中 50 个 AUC 上升、26 个 AUC 下降。按
76 个指标等权平均，AUC 差值为 +0.002115，中位数为 +0.001452；最大的单项提升和
下降分别为 +0.030859 和 -0.040123。Logloss 在 51/76 个指标上改善，平均差值为
-0.002229。若对 16 个“模型 × 数据集”单元等权平均，AUC 宏平均差值为 +0.001349，
其中 12/16 个单元为正。

| 模型 | 数据集 | 平均 AUC 差值 | AUC 提升标签数 | 平均 Logloss 差值 |
|---|---|---:|---:|---:|
| OneTrans | QK-Video | +0.000437 | 3/4 | -0.000193 |
| OneTrans | KuaiRand | +0.007690 | 5/6 | -0.008736 |
| OneTrans | Taobao | -0.003796 | 1/4 | +0.000127 |
| OneTrans | MerRec | +0.009145 | 4/5 | -0.001963 |
| HiFormer | QK-Video | -0.002100 | 1/4 | -0.000029 |
| HiFormer | KuaiRand | +0.003684 | 3/6 | -0.003161 |
| HiFormer | Taobao | -0.025167 | 0/4 | +0.000411 |
| HiFormer | MerRec | +0.001194 | 3/5 | -0.001831 |
| RankMixer | QK-Video | +0.001231 | 4/4 | -0.000764 |
| RankMixer | KuaiRand | +0.008597 | 6/6 | -0.004843 |
| RankMixer | Taobao | -0.000596 | 0/4 | +0.000016 |
| RankMixer | MerRec | +0.007506 | 4/5 | -0.002186 |
| Zenith | QK-Video | +0.000736 | 4/4 | -0.000664 |
| Zenith | KuaiRand | +0.006733 | 6/6 | -0.004647 |
| Zenith | Taobao | +0.003102 | 3/4 | -0.000028 |
| Zenith | MerRec | +0.003184 | 3/5 | -0.001337 |

因此，SISA 整体上带来正向收益，但效果明显依赖模型和数据集。KuaiRand 与 MerRec 在
四个模型上的单元均值均为正；QK-Video 整体接近中性；Taobao 在三个模型上为负，
仅 Zenith 为正，其中 HiFormer-Taobao 是最明显的退化。严格协议只运行论文匹配的一个
seed 和一个 epoch，因此这些配对结果是点估计，不是置信区间，也不能证明跨 seed 的
统计显著性。

能找到作者 benchmark 日志的 baseline 均已直接进行复现对比。仓库没有
RankMixer-KuaiRand 和 RankMixer-MerRec 的对应 base 日志，因此这两组使用论文表 3
保留四位小数的结果。下表给出每个单元内最大的 AUC 绝对差值：

| 模型 | QK-Video | KuaiRand | Taobao | MerRec |
|---|---:|---:|---:|---:|
| OneTrans | 0.001302 | 0.011582 | 0.003624 | 0.018662 |
| HiFormer | 0.001388 | 0.007278 | 0.009146 | 0.005349 |
| RankMixer | 0.000237 | 0.001198* | 0.001874 | 0.001803* |
| Zenith | 0.000185 | 0.002697 | 0.000957 | 0.005044 |

`*` 表示与论文四位小数结果比较。全部 76 个 baseline AUC 的平均绝对差值为
0.002535，其中 72/76 在 0.01 内。四个较大差值为 OneTrans-MerRec Cart
（-0.018662）、OneTrans-MerRec Offer（+0.012236）、OneTrans-KuaiRand Follow
（+0.011582）和 OneTrans-KuaiRand Forward（+0.010106）。前述协议、数据、DDP、
优化器/更新次数、checkpoint 和 exact-off 审计均未发现实现不一致，因此这些值作为
跨硬件/软件栈的复现波动如实保留，没有被静默删除。

该实验完成时，机器可读证据存放在 HPC01 上 Git 忽略的
`artifacts/sisa_native_strict/` 目录。2026-08-25 已按精确迁移清单复制到 HPC3，并将
适合版本控制的小型结果归档到本报告旁的 [`results/`](results/)：

- [`results/runs.csv`](results/runs.csv)：32 个逻辑任务的 job/task ID、GPU 型号、协议、完成标记、指标有限性、
  错误数和完成状态；
- [`results/metrics.csv`](results/metrics.csv)：全部 152 行 baseline/SISA 原始指标；
- [`results/summary.md`](results/summary.md)：全部 76 组配对 AUC/Logloss 及其差值。

迁移包、32 份 Slurm 日志、32 份训练指标日志和离线可视化报告的说明见
[results/README.md](results/README.md)。迁移后已执行逐文件存在性、rsync checksum 和
CSV 语义复核，未发现缺失或内容差异。

2026-08-24 最终复跑的 29 个回归测试全部通过：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

严格矩阵提交时的 runtime fingerprint 为：

```text
9cf06ce48728f338caa8c07b12d0dd38596b250c8e18692c96aae6ec7fdf507c
```

最终 SISA 重试任务 `14846` 所使用四个文件的 runtime fingerprint 为：

```text
799f22848f96080106b99e5e96dd7f78cf9d159fd9fc279a4b44f1fb5a7303a7
```

## 监控与最终证据

后台监控器每次采样后 `sleep 1800` 秒，只向 Git 忽略的
`logs/sisa-native-monitor.log` 追加内容。它从 02:33 持续记录到 23:52 的最终采样，
内容包括 Slurm 状态、训练进度、错误模式、GPU 利用率/显存、存储和数据状态；随后因
没有可运行实验或下载进程而正常退出。绝大多数采样间隔约为 30 分钟。17:14 至 18:48
期间，监控输出重定向问题造成文件记录缺口；该时段通过终端直接巡检覆盖，重启监控器
时没有中断任何训练任务。

探索性单卡结果与严格实验结果必须始终使用不同的结果表。对于每个严格逻辑任务，最终
记录包括模型、数据集、baseline/SISA 设置、GPU 型号、Slurm job/task ID、ExitCode、
测试指标和完成标记状态。HPC01 可能在下一次 30 分钟采样前从 `scontrol` 中清理已经
完成的任务；此时收集器明确记录 `COMPLETED_MARKER / 0:0 (success marker)`。该标记
只会在 `set -euo pipefail` 下的 `torchrun` 成功返回后写入。如果实时 Slurm 状态和
ExitCode 仍然可用，则优先使用它们。当前 32 个严格任务已经全部通过这些检查。

任务 10/11、26/27 和 30/31 选择自重试数组，因此应使用以下命令重新生成最终矩阵：

```bash
.venv/bin/python scripts/collect_sisa_native_strict_results.py \
  --job-override 10=14850 \
  --job-override 11=14850 \
  --job-override 26=14854 \
  --job-override 27=14854 \
  --job-override 30=14843 \
  --job-override 31=14846 \
  --gpu-type-override 10=l40s \
  --gpu-type-override 11=l40s \
  --gpu-type-override 26=l40s \
  --gpu-type-override 27=l40s \
  --gpu-type-override 30=l40s \
  --gpu-type-override 31=l40s
```

预期终端输出为：

```text
strict_complete=32/32 output_dir=artifacts/sisa_native_strict
```
