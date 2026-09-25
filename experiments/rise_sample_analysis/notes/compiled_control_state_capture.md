# 逐variant的compiled内部控制状态捕获：失败证据和新方案

日期：2026-09-23。本记录继承既有失败/成功回执，不修改冻结的observer、diagnosis、benchmark、controls或core源文件。

## 1. 643911的SIGSEGV只能有限定位

调度器确认643911为 `FAILED 11:0`，用时93秒，节点ACD1-63；MaxRSS未返回。日志只有Slurm shell报告的 `Segmentation fault (core dumped)`，没有Python或C栈。因此，不能凭此断言dynamo.reset、GPU释放、模型析构或内存不足中的任何一项是根因。

B已完整写出512行5标签的四路预测和3,072条validated selection。B capture/plain、与643858 compiled参照均bitwise相同；eager旧hook/plain也bitwise相同。E没有结果文件。可定位的窗口是B结果写出之后、E结果产出之前；不能进一步把精确崩溃位置说成“加载E时”而无栈证据。

当前安全的执行策略是保留完整B证据，只补缺少的E，使用单variant进程与新输出目录，并打开 `PYTHONFAULTHANDLER=1`/`-X faulthandler`。这是隔离编译和对象生命周期的运行策略，不是经过证实的根因修复。

```bash
PYTHONFAULTHANDLER=1 .venv/bin/python -X faulthandler \
  experiments/rise_sample_analysis/scripts/diagnose_mechanism_precision.py \
  --failed-dir artifacts/rise_sample_analysis/mechanism/observed_643858 \
  --output-dir artifacts/rise_sample_analysis/mechanism/precision_E_JOBID \
  --variants E --limit-batches 8
```

整体643911作业仍是失败状态；独立通过验证的B子结果与缺失的E必须分开记录，不追改调度器事实。

## 2. 643918说明eager重算不能代替E的compiled控制

64个相同输入的逐位置审计结果：

| Variant | Δ位置数 | 失败位置数 | 最大Δ绝对误差 | 最大selection TV差异 | 结论 |
|---|---:|---:|---:|---:|---|
| B content-only | 3,916,800 | 0 | 0 | 0 | 重算精确一致 |
| E full | 3,916,800 | 1,801,700 | 0.3125 | 0.0265378 | Δ与selection两道门槛均失败 |

E六个站点均失败，21,440/32,658个有效attention rows超过selection TV门槛。两variant的compiled capture最终预测仍通过原门槛。由此可以说：对这些固定compiled表示，原eager执行的完整SISA重算不足以数值复现E的实际Δ；不能采用其AM/PA重算统计作为实际compiled机制。B完全一致帮助排除“所有辅助内容投影都必然无法重算”的过度概括，但还不能单独把E差异唯一归于AM或PA中的某一个。

门槛未放宽；643918保留为数值方法审计结果。其成功退出仅意味着审计运行完整，不代表E重算通过。

## 3. 新的直接状态捕获入口

`SISAScoreBias._final_channels` 接收到的四个参数已经是当前graph算出的 `raw_channels`、decay state、phase state和centering offset，其输出是进入head scale之前的辅助channels。因此可以在实例上临时包装这个bound method：记录其参数和返回tensor引用，再返回原方法的原输出，不另做从hidden出发的投影/scan重算。

新增 `scripts/capture_compiled_control_states.py`：

1. 正常加载原compiled checkpoint并获得无instrumentation预测。
2. 仅在当前模型实例上包装 `_final_channels`，同时沿用Q/K/Δ tensor-only capture；记录tensor引用时不做运算、CPU复制或NumPy操作。
3. 清理Dynamo cache后重编译，执行同一个固定小批。
4. forward结束后才从捕获状态推导clipping、AM幅度和phase统计；直接从原/最终channel计算norm gain、cosine变化和符号变化。
5. 对比无instrumentation预测及643858冻结compiled预测，继续使用原概率门槛；同时显式报告bitwise equality。
6. 只有gate通过且样本/站点完整，才产 `*_compiled_states_validated.parquet` 和 `complete.json`；否则保留unvalidated证据并失败退出。

程序只允许一次一个variant，启用faulthandler并持续写阶段 `progress.json`，以便若再出现native crash可以缩小发生窗口。

## 4. 这些统计是什么、又不能被怎样命名

直接捕获的是 **instrumented compiled graph的实际method输入/输出tensor**，最终模型预测是否保持不变由独立gate验证。它不是eager从hidden重算的控制，也不声称读取到了未经物化的GPU kernel物理寄存器。

- `state_derived_query/key_clipping_fraction`：由捕获decay state、offset及原clip bound推导的命中比例。
- `query/key_phase_state_abs_mean`：捕获的graph phase state幅度；并非从输入重新投影。
- `state_derived_log_amplitude/amplitude_mean`：由捕获state按原幅度定义推导。
- `captured_channel_norm_gain_mean`：直接比较 `_final_channels` 输入raw channels和实际返回channels的范数变化，包含最终low-precision舍入的影响。
- `captured_channel_cosine_change_abs_mean` 和 alignment sign change：直接比较变换前后channel的方向，不需要重新旋转向量。
- 原actual Q/K/Δ的Ω、TV、padding质量等selection统计同时保留。

这些是固定表示、固定权重的条件观测，不等于重新训练关闭某一模块后的性能增量。就算instrumentation预测bitwise不变，也不应声称每个未观测硬件临时值都与原kernel逐位相同；说明graph-level捕获边界即可。

## 5. CPU证据与下一次短GPU命令

`tests/test_compiled_control_states.py` 的2项CPU测试通过，主体用时3.160秒：六站点full在连续两次compiled forward中各捕获12次 `_final_channels`，输出与原模型bitwise一致，退出后恢复原实例方法；单站点local target-attention也连续两次保持bitwise，并直接捕获到零gauge。CPU backend为Dynamo eager，尚不能代替GPU Inductor gate。

新类还处理了一个可复現的结构细节：`OptimizedModule` 会转发其内部module属性；遍历named_modules时必须跳过wrapper本身，才能避免对被单独compile的target-attention重复注册同一site。这一处理只在新类中实现，未改冻结诊断脚本。

```bash
.venv/bin/python experiments/rise_sample_analysis/scripts/capture_compiled_control_states.py \
  --reference-dir artifacts/rise_sample_analysis/mechanism/observed_643858 \
  --output-dir artifacts/rise_sample_analysis/mechanism/states_E_probe_JOBID \
  --variant E --limit-batches 1
```

先验收E的64行GPU gate；成功后同脚本 `--limit-batches 8` 覆盖512行。需要B时另起一个进程/输出目录。CPU原型已支持单站点local模块，所以若此捕获方式在GPU上保持预测不变，随后可扩展到RankMixer E/L实权重，检验local零gauge与full居中方式的clipping差异是否在实际样本上激活；当前HyFormer CLI尚未假装支持RankMixer数据或已完成其GPU检查。

本子任务未提交GPU，旧失败文件和旧源码均保留。
