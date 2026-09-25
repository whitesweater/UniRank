# 原生A、Content-only B与完整RISE E：扩展成本实验冻结方案

日期：2026-09-23。本研究使用已经完成的HyFormer–MerRec权重，只测推理成本，不新增训练。A为独立训练的原生baseline，任务643809_0；B为组件content-only，E为完整RISE。**正式原生开销是E/A，不能用E/B冒充。**

## 1. 实际输入缓存已经构建并验证

来源为 `artifacts/rise_sample_analysis/mechanism/observed_643858/sample_batches.pt` 中512个真实诊断样本。源文件SHA256、原manifest、原row table及其一致性均已检验。构建过程中没有重复样本、填充样本数量、重新采样、依据标签/预测挑选或修改张量。

新缓存：`artifacts/rise_sample_analysis/mechanism/benchmark_cache_643858/benchmark_batches.pt`。

| sample-index | 行数 | 来源 | 每entry身份和张量校验 |
|---:|---:|---|---|
| 0 | 64 | 原cache第一项，part2 | 唯一row IDs；与旧64行输入逐tensor相同 |
| 1 | 256 | 原cache前四项，part2/3/4/5各64 | 唯一row IDs；按原顺序拼接 |
| 2 | 512 | 原cache全部八项，part0–7各64 | 唯一row IDs；覆盖全部原诊断输入 |

不同entry是有意嵌套的前缀，因此entry之间会重叠；每个entry内部没有重复row IDs。改变batch大小同时改变这个固定样本前缀的内容分布，所以它是“这些真实输入上的batch-size成本观察”，不是完全固定分布的总体效率定律。

`manifest.json` 保存源文件/manifest/row table哈希、每entry的 `identity_sha256`、`tensor_tree_sha256`、part分布和冻结benchmark脚本哈希；`entry_rows.parquet` 保存每entry每行的完整来源身份。`complete.json` 只在写出后重新加载并逐entry核验成功后生成。

实际验证：新的64行entry身份哈希为 `15c314cf4996caafef7ac54f64687e37724a0c74fb0562a331a4e38af18268bb`，与643870正反两轮的B/E四份记录全部一致；张量内容也与原cache第一项完全一致。旧日志因此保有可追溯输入联系，但旧warmup30与新warmup1000不同，不能直接混为同一计时协议的重复测量。

## 2. 原生A协议核对

原生日志/权重：

`artifacts/rise_sample_analysis/training_checkpoints/MerRec_Action/RISE_matched_HyFormer_MerRec_Action_baseline_seed20262028_attempt1.{log,model}`

A通过现有benchmark使用的 `validate_parameters(...,'A')` 全字段校验。与E差异仅包括模型输出名称、SISA关闭和已允许的checkpoint输出目录；其数据与训练协议不发生改变。新成本任务直接加载A自身的checkpoint，不以E临时关闭模块替代A。

`benchmark_inference.py` 保持未修改，其SHA256与643870 manifest一致。新builder和runner把该SHA256纳入冻结与运行前验证。

## 3. 精确任务矩阵与资源

| 批大小 | round 1顺序 | round 2顺序 | 每variant warmup | 实际计时 |
|---:|---|---|---:|---|
| 64 | A→B→E | E→B→A | 1,000 forwards | 10 blocks × 100 forwards |
| 256 | A→B→E | E→B→A | 1,000 forwards | 10 blocks × 100 forwards |
| 512 | A→B→E | E→B→A | 1,000 forwards | 10 blocks × 100 forwards |

共6个round、18个独立variant子进程。每个variant模型单独加载，固定compiled、eval、no_grad、BF16；数据先在GPU，计时排除loader、H2D、权重读取、初次compile和warmup。使用现有CUDA event block计时、显存峰值、kernel profiler及环境快照实现。

Slurm配置：debug、1节点、1GPU、4CPU、120GiB主机内存、30分钟。30分钟是本次资源预算与预计，不是已测执行时间；实际由逐round、orchestrator和shell回执记录，并以Slurm accounting核对。

## 4. 新增执行文件

- `scripts/build_benchmark_cache.py`：构建或 `--verify-only` 验证上述缓存，禁止覆盖。
- `scripts/run_extended_benchmark.py`：校验缓存、冻结benchmark源码及A协议，调用原benchmark六次，保留每round完整结果和wall time。
- `scripts/submit_extended_benchmark.sbatch`：指定资源，复用已准备缓存，捕获shell实际用时/退出状态。
- `tests/test_benchmark_cache.py`：缓存身份和张量结构的CPU回归。

以上相对路径均在 `experiments/rise_sample_analysis/`。本子任务没有提交GPU。

正式提交命令：

```bash
sbatch experiments/rise_sample_analysis/scripts/submit_extended_benchmark.sbatch
```

预计结果目录：`artifacts/rise_sample_analysis/mechanism/extended_benchmark_<jobid>/`，内含 `batch64_forward01`、`batch64_reverse01`、`batch256_forward01`、`batch256_reverse01`、`batch512_forward01`、`batch512_reverse01`。每个子目录仍是原benchmark格式，保留A/B/E各自的10个原始block均值、绝对延迟、显存、kernel和环境快照。

## 5. 用时、完成证据与不完整运行

`round_receipts.json` 记录每round包含加载、编译、warmup、计时和profile的实际wall seconds；`execution_outcome.json` 记录orchestrator总时间和成功状态；`job_shell_outcome.json` 记录shell跨度与退出状态。上述wall seconds不能替代benchmark中的GPU forward latency。

只有6轮全部完成、18模型记录齐全、各模型输入identity匹配预期entry、warmup1000/blocks10/repeats100一致、同一个GPU UUID时，根目录才生成 `complete.json`。某轮失败或超时保留此前原始block与失败回执；不能只整理成功/较快的轮次宣布全研究完成。Slurm强制终止可能早于所有finally/trap执行，因此调度器terminal状态和缺失receipt也必须如实纳入验收。

## 6. 汇报规则

每个batch同时展示正反两轮A、B、E的绝对延迟与显存，分别给出E/A和E/B，保留原始block分布。若给一个压缩指标，可以固定使用两轮median的中位数，并同时报告两轮范围；不得选择最优一轮作为正式开销。

643870旧64行实测已说明顺序/环境波动不可忽略：E/B两轮分别约1.0007和1.0645。这两轮只包含B/E，不包含原生A，不能支持RISE相对原生模型几乎无开销。它们继续作为旧协议记录保存，不因新结果产生而删除。

CPU环境、GPU功耗/时钟、其他GPU进程与profiler路径需逐轮检查。block不是独立训练seed，block分位数不是请求tail latency，单GPUresident forward不是含数据管道的端到端吞吐。

## 7. 本轮实际准备验证

6项缓存CPU tests全部通过，主体用时0.016秒；覆盖嵌套前缀逐tensor与row身份、拒绝重复样本放大batch、拒绝超出真实样本量、shape/dtype一致性、内容hash敏感性及BF16 hash支持、metadata dtype漂移导致大整数ID舍入的防护。`bash -n`、`py_compile`、`git diff --check`通过。

runner的 `--dry-run` 已实际通过，返回缓存验证成功、A协议成功、精确6轮和18个模型进程。实际缓存已生成并重新读取验证；GPU成本测量尚待主agent提交和验收。
