# HyFormer–MerRec 维度实验提交记录

北京时间 2026-09-23 02:15:51 提交 Slurm 数组 643736。

| 数组任务 | 每头 SISA 维度 | attempt | 实验 seed |
|---|---:|---:|---:|
| 643736_0 | 8 | 1 | 20262028 |
| 643736_1 | 32 | 1 | 20262028 |
| 643736_2 | 64 | 1 | 20262028 |

16 维复用已经重新审计的完整 SISA 任务 548166_1，不新增训练。
用户已排除 4 维，范围仅限 HyFormer–MerRec。

提交命令：

```bash
sbatch --parsable scripts/submit_sisa_score_dim.sbatch
```

每个任务请求 acd_u 分区两张 GPU、16 CPU、480 GiB 主机内存、12 小时上限，
数组并发上限为 2。保持全局 batch 32768、每卡 batch 16384、1 epoch、
max_len=100、loader seed=2027 和 SISA parameter seed=20260822。

提交前完成的检查：

- 当前配置与冻结历史对照的 225 项字段比较通过，差异仅为 score_dim 和 model_id。
- 当前 train/valid/test 在两个 rank 上的块数、行数、批次数、保留行数与历史一致。
- 原始 checkpoint/Slurm 最终 Test、五标签覆盖及归档值一致；16 维宏平均
  AUC=0.8008724、LogLoss=0.0826776。
- 完整 CPU 回归：141 tests，20.240 秒，OK。
- Slurm 启动脚本与守护脚本分别通过 bash -n。
- 三个任务的输出路径不存在，启动器会拒绝覆盖已有日志、模型和运行清单。

精确提交回执位于本目录 submission.json，预检记录位于 preflight.json，
训练源码与冻结配置散列位于 source_sha256.json。新脚本尚未提交 Git，
运行时同时记录 Git revision、工作区状态及逐文件散列，不能仅以 Git revision
推断运行源码。实验过程中不得修改已冻结的训练文件或配置。

北京时间 02:16:18 已启动守护，初次检查三个任务均为 PENDING/Priority。
每轮采集后执行 sleep 600，再次核对队列、训练进度和结果审计状态。
守护启动回执位于 monitor.json；完整快照位于 artifacts/sisa_score_dim/monitor/。
当前 results/ 记录未完成状态，不是新维度的实验结果。

重新采集进度与结果的命令：

```bash
.venv/bin/python scripts/collect_sisa_score_dim_results.py
```

运行中的同一研究只允许一个守护实例，脚本使用 flock 防止重复。若守护已经结束且
需要恢复，可运行 bash scripts/monitor_sisa_score_dim.sh；不要重复提交训练数组。

## 运行中记录

02:46 左右，8 维与 32 维任务开始运行，分别分配 ACD1-24、ACD1-13，
均为两张 NVIDIA H100 80GB HBM3。运行时 65 项训练源码和冻结配置散列
与提交前快照一致，实际环境记录于各自的 manifest.json。

02:56 的 Slurm 会计输出同时包含 32 维的实际运行记录（JobIDRaw=643791）
和数组父任务展开的旧 PENDING 占位记录（JobIDRaw=643736）。采集器最初
被后出现的占位记录覆盖，误写 PENDING 并产生 job_id 校验问题。
03:00 已修复采集器：具体任务记录优先于父数组的 PENDING 占位记录；
新增回归用例覆盖两种行顺序以及完成记录。七项维度研究测试通过，
修复后的独立采集显示 8 维和 32 维均为 RUNNING。该问题只影响状态采集，
训练源码、配置和任务均未改动。原始检查快照保留。

03:26 守护检查时，8 维和 32 维均为 COMPLETED/0:0，全部证据审计通过。
64 维任务于 03:23 左右失败，Slurm 状态 FAILED/1:0，耗时 00:02:10，
分配节点 ACD1-2。两个 rank 在首个编译训练前向均 OOM，未完成一个训练批次。
守护以 NEEDS_ATTENTION 结束，保存当轮快照。

当前队列仅暴露 ACD 资源，未发现可直接选择的更大显存节点。
已检查现有非重入激活检查点方案：在 r64 HyFormerBlock 上，FP32、BF16
的输出、输入梯度以及 132 个有效参数梯度均与不开检查点逐元素完全一致。
这只是 CPU 检查，尚未证明同 batch 的 GPU 峰值显存可行。
下一步等待选择：保持原 batch、seed 和数据覆盖开启检查点重试，或保留 OOM
作为本协议的资源边界。不擅自缩小 batch、增加梯度累积或更改训练数据覆盖。

## 64 维的独立重试

继续完成全部实验的指令后，采用已有非重入激活检查点方案处理 OOM。
03:45:11 仅提交 task 2、attempt 2，实际任务为 643901_2。8、32 维不重跑，
16 维仍复用原对照。重试使用隔离 YAML 配置，训练骨干和 SISA 实现未改动。
与完整对照的有效参数差异只有 sisa_score_dim、model_id、gradient_checkpointing。
batch、world size、梯度累积、种子、优化器和数据划分保持一致。

重试提交前 145 项 CPU 回归测试全部通过，其中新增检查点输出/梯度等价测试、
重试配置边界测试，以及带执行差异标签的结果接纳测试。
完整回执位于 submission_r64_attempt2.json，当前选择位于 submission.json，
原始提交回执保存在 submission_attempt1.json。
第一轮的 65 项训练源码与配置已保存到 artifacts/sisa_score_dim/attempt1_source/；
新一轮散列位于 source_sha256_r64_attempt2.json，按运行分别匹配。

初次失败仍保存在 attempt_history.json、r64_attempt1_failure.json 和原始日志中。
同一守护脚本已重启，当前 PID 见 monitor.json，第一轮守护回执保存在
monitor_attempt1.json。只采集本研究的数组，继续每轮 sleep 600。

## 编译显存诊断与最终四卡方案

643901_2 在 03:58:37 启动，04:00:48 以 FAILED/1:0 结束；外层检查点仍未
避免首个编译前向 OOM，失败记录保存在 r64_attempt2_failure.json。
随后只进行了短诊断，没有额外训练对照或生成正式 Test 指标：

- 643933：两 rank 的小批量 GPU 138 张量数值比较完全一致；诊断 hook 的类名导入错误导致退出，随后修正。
- 643996：整块编译的 activation_memory_budget=0.0 通过前向，首个反向 OOM。
- 644025：分层编译没有通过 BF16 数值接近性检查，没有进入全 batch 训练步。
- 644051：整块编译的 activation_memory_budget=0.5 通过数值检查和三个原 batch DDP 训练步，
  两 rank 最大绝对数值误差为 0，峰值已分配显存约 76.76 GiB。未继续扫描其他预算。

这些检查的回执和结果位于 memory_probe_*.json，代码版本及输出另行归档。
用户随后明确要求增加 GPU 数、减小单卡 batch、保持全局 batch；最终重试因此
采用四卡方案，未将编译器显存预算方案用于正式结果。

10:42:30 已提交 644583_2（attempt 3），请求 4 张 H100、32 CPU、480 GiB 主机内存、
2 小时时限；每卡 batch=8192、global batch=32768、accumulation_steps=1。
使用原 run_expid.py、原整块编译与默认编译器设置，gradient_checkpointing=False。
所有模型、数据配置、优化器及种子保持，允许的日志参数差异为维度、运行名、world_size、batch_size。
151 项 CPU 测试全部通过，四卡任务只包含这一个未完成维度。

四卡下的 drop_last 保留行数预检：训练 144,072,704，验证 16,482,304，测试 11,640,832；
相对两卡分别增加 24,576、32,768、40,960。分片变化及样本数差异必须写入最终报告。
每个 rank 的预检覆盖冻结于 configs/data_coverage_ws4.csv；有效参数比较位于
r64_attempt3_protocol.csv，精确回执位于 submission_r64_attempt3.json。
启动后已核对实际分配为四张 NVIDIA H100 80GB HBM3，源码和冻结配置散列与提交前一致。
第三轮守护 PID 见 monitor.json，前两轮回执均保留。

10:52 守护检查：四卡训练约 38%，实际 world_size=4、per_gpu_batch=8192、global_batch=32768，
未发现协议、源码或参数量问题。11:02 的采集因四 rank 并发进度条造成的无效 UTF-8 字节失败，
作业仍在正常训练。随后只修复本研究的日志读取：Slurm .out 以原始 bytes 计算 SHA-256，
对破损的非 ASCII 字符作替换显示并记录数量；配置 JSON 继续严格解码，正式指标仍严格解析。
新增回归测试核对原始散列和 ASCII Test 指标不受影响，13 项维度研究测试通过。
11:04:45 的独立采集已确认训练结束并进入验证，未发现训练异常。

## 完成与最终审计

644583_2 于 11:07:16 以 COMPLETED/0:0 结束，耗时 00:24:21。
11:13:15 的最终守护快照中，8、16、32、64 维全部 eligible=True，守护随后正常结束。
64 维最终宏平均 AUC=0.8044016、LogLoss=0.0816382；相对 16 维分别为
+0.0035292 和 -0.0010394。仅 Offer 的 AUC 高于 16 维，其余四标签 AUC 略降。

四个维度的 40 个标签级指标已从原始两份日志独立核验，所有模型文件均保留。
最终完整 CPU 回归 152 项通过，Slurm/守护脚本语法检查通过。
两卡/四卡覆盖差异、两次正式失败、四次诊断及后续处理均已保留在独立报告和归档中。
生成器输出写入 archive/reports/generated_report.md，正文首版单独复制为 report.md，
后续自动生成不覆盖可手工编辑的正文。
