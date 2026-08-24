<p align="center">
  <img src="./assets/figures/unirank_logo.png" alt="UniRank logo" width="720">
</p>

# UniRank 项目说明与实验交接

> 最后更新：2026-08-24。本仓库是 UniRank 的可复现研究分支，同时包含 SISA
> 适配、严格配对实验、集群运行脚本和审计工具。本文既是项目入口，也是后续实验的
> 唯一主交接文档。

## 1. 项目是什么

UniRank 是面向大规模推荐排序的 PyTorch 基准框架，统一了时序行为建模、非时序特征
交互、多反馈任务、blocked Parquet 数据加载、DDP 训练和分布式评测。本分支基于上游
`v0.7.2`/`90bb99a`，主要目标是在不改变基线主体结构和训练协议的前提下，评估 SISA
对原生交互分数的增量效果。

仓库包含 15 个排序模型：

`HiFormer`、`RankMixer`、`INFNet`、`LONGER`、`OneTrans`、`Zenith`、
`HyFormer`、`MixFormer`、`TokenMixer`、`EST`、`HeMix`、`UniMixer`、
`TokenFormer`、`UltraHSTU` 和 `SSR`。

核心训练范式是 chronological point-wise autoregressive supervision：每个合格的
时间位置都可以成为监督目标，只使用该位置之前的因果历史；验证和测试同样遵循时间
顺序。不同模型共享数据、任务、指标和训练入口，模型差异集中在交互骨干。

## 2. 当前状态总览

| 工作项 | 状态 | 说明 |
|---|---|---|
| 上游 UniRank 基线 | 已接入 | 上游基点为 `90bb99a` |
| SISA 公共实现 | 已完成 | `unirank/pytorch/layers/attentions/sisa.py` |
| 原四模型 SISA | 已完成 | OneTrans、HiFormer、RankMixer、Zenith |
| 原严格矩阵 | 已完成并审计 | 4 模型 × 4 数据集 × baseline/SISA，共 32 个任务 |
| TencentGR 补充矩阵 | 待运行 | 原四模型 × TencentGR × baseline/SISA，共 8 个任务 |
| 新三模型 SISA | 已完成代码与测试 | UniMixer、HyFormer、UltraHSTU |
| 新三模型五数据集矩阵 | 待运行 | 3 模型 × 5 数据集 × baseline/SISA，共 30 个任务 |
| 扩展启动器 | 已完成 | 统一包含上述 38 个待运行任务 |
| HPC01 L40S 串行运行 | 已提交、等待资源 | allocation `14984`；4×L40S；获批后串行并支持跨 allocation 续跑 |
| 本地回归测试 | 已通过 | 当前 35 个测试全部通过 |

最重要的交接边界：**32 个原严格任务已经完成；38 个扩展任务已经提交，但尚未取得
L40S 资源和正式结果。** 不要把排队、适配器实现完成或 smoke test 通过写成正式实验完成。

## 3. 仓库与集群位置

| 位置 | 路径/地址 | 用途 |
|---|---|---|
| 个人 GitHub | `https://github.com/whitesweater/UniRank` | 私有主远端，`main` 是同步源 |
| 上游 GitHub | `https://github.com/salmon1802/UniRank` | 只用于跟踪官方更新，远端名 `upstream` |
| HPC01 | `/data_nvme/user/ywhao/proj/UniRank` | 已完成的严格实验、审计与当前开发目录 |
| HPC3 | `/data/user/yhao481/proj/UniRank` | 第二份运行 checkout，SSH 别名 `hpc3_27` |

Git 只同步实现和复现必需文件。以下内容按设计不进入仓库：

- `datasets/`：预处理数据和软链接；
- `logs/`、`checkpoints/`：训练日志和模型；
- `artifacts/`：结果收集器生成的机器可读产物；
- `.venv/`、编译缓存和临时目录；
- 本机专用的中间文件。

因此，“代码三端一致”不等于“数据和历史日志已经复制”。在新的集群运行前必须单独
核对数据路径、Slurm 资源、Python/CUDA 环境和输出目录。

## 4. 目录和关键入口

```text
UniRank/
├── config/
│   ├── dataset_config.yaml          # 数据路径、schema、label、vocab
│   └── model_config.yaml            # 模型/数据集实验配置
├── model_zoo/                       # 15 个模型及 SISA 适配位置
├── unirank/                         # dataloader、训练、指标和公共层
├── data/                            # 原始数据预处理脚本
├── benchmark/                       # 上游参考日志
├── scripts/
│   ├── submit_sisa_native_strict.sbatch
│   ├── submit_sisa_expansion.sbatch
│   ├── request_sisa_expansion_l40_all.sh
│   ├── run_sisa_expansion_l40_all.sh
│   ├── supervise_sisa_expansion_l40.sh
│   ├── monitor_sisa_expansion_l40.sh
│   ├── submit_sisa_native_smoke.sbatch
│   ├── unirank_gpu_smoke.py
│   └── collect_sisa_native_strict_results.py
├── tests/                           # 回归、协议和结果审计测试
├── run_expid.py                     # 单实验/DDP 主入口
├── run_all.sh                       # 批量入口
├── SISA_STRICT_EXPERIMENTS.md       # 已完成 32-task 的完整证据
├── pyproject.toml / uv.lock         # 锁定环境
└── README.md                        # 项目说明与主交接文档
```

## 5. 数据集

| 配置 ID | 场景 | 标签数 | 主要反馈 |
|---|---|---:|---|
| `QK_Video_Action` | 短视频 | 4 | click、follow、like、share |
| `KuaiRand_Video_Action` | 短视频 | 6 | click、follow、like、comment、forward、long-view |
| `TencentGR_10M_Action` | 广告/推荐 | 2 | click、conversion |
| `Taobao_Action` | 电商广告 | 4 | click、cart、favorite、buy |
| `MerRec_Action` | 电商 | 5 | like、cart、offer、checkout、purchase |

论文中的 TAAC-25 在当前配置中对应 `TencentGR_10M_Action`。预处理数据应放在
`./datasets/<dataset_id>`；`config/dataset_config.yaml` 使用仓库相对路径，避免把某个
集群的绝对路径提交到 Git。

blocked 数据按 split 保存匹配的三类分块：

```text
datasets/<dataset_id>/train/
├── data/part-*.parquet
├── user_info/part-*.parquet
└── item_info/part-*.parquet
```

作者发布的数据已经编码完成，`rebuild_dataset: False` 时直接根据 YAML 中的
`vocab_size` 构建只读 FeatureMap，不生成数百万条目的 `feature_vocab.json`。

## 6. SISA 实现说明

### 6.1 公共约束

SISA 公共模块生成带衰减和旋转的可学习分数通道。每头正权重 `lambda_h` 通过
`softplus` 参数化；B/C、decay 和 phase 投影均可学习。

所有适配器必须同时满足：

1. `sisa_enabled: false` 时不注册任何 SISA 参数，基线初始化和输出路径保持不变；
2. `sisa_enabled: true` 且 `sisa_score_scale: 0` 时是逐元素精确的零扰动对照；
3. 正常启用后，SISA 参数必须获得有限且非零梯度；
4. 不能重新打开原模型 hard mask，也不能替换基线 loss、tower 或训练协议；
5. 每个站点使用隔离的确定性参数 seed，不消耗基线的全局初始化 RNG 流。

### 6.2 七个已支持模型

| 模型 | SISA 接入位置 |
|---|---|
| OneTrans | 原生 self-attention pre-softmax score |
| HiFormer | 原生序列 attention |
| RankMixer | 仅已有 target-attention pooling；不修改 token mixer |
| Zenith | 原生 attention score |
| HyFormer | 每层序列 self-attention + query-to-sequence cross-attention |
| UltraHSTU | 将等价 SISA 通道拼接到 FlexAttention Q/K，保留 sparse block mask |
| UniMixer | pre-Sinkhorn global-mixing logits；这是 mixer adapter，不宣称为 attention |

UltraHSTU 不使用捕获外部张量的 `score_mod`。当前实现通过 Q/K 扩展得到与 additive
SISA bias 数学等价的分数，同时避免构造 `[B,H,S,S]` 稠密 bias。UniMixer 的
`use_tau_symmetry` 约束应用于加入 SISA 后的完整 global logits。

## 7. 已完成的 32-task 严格实验

原严格矩阵为：

- 模型：OneTrans、HiFormer、RankMixer、Zenith；
- 数据集：QK-Video、KuaiRand、Taobao、MerRec；
- 设置：baseline 与 SISA；
- 总任务数：`4 × 4 × 2 = 32`。

统一协议：

- 四张同型号 GPU，`torchrun --nproc_per_node=4`；
- 每卡 batch 8192，global batch 32768；
- accumulation 1，训练 1 epoch，seed 20262027；
- `max_len=100`，BF16，blocked DDP；
- baseline/SISA 配对使用同型号 GPU；
- 成功任务必须写入 `SISA_NATIVE_STRICT_COMPLETE`。

最终审计结果：32/32 完成，76 个标签级配对指标中 50 个 AUC 上升、26 个下降，平均
AUC 差值 `+0.002115`；按 16 个模型/数据集单元等权，宏平均差值 `+0.001349`，其中
12/16 个单元为正。该矩阵只使用一个固定 seed，因此是受控预算下的点估计，不是跨
seed 的统计显著性结论。

完整 job ID、重试、硬件配对、OOM 兼容处理、baseline 复现偏差、逐单元结果和证据
生成方式见 [SISA_STRICT_EXPERIMENTS.md](SISA_STRICT_EXPERIMENTS.md)。历史 pilot 为
单卡 global batch 8192，只作诊断，禁止混入严格结果表。

## 8. 待运行的 38-task 扩展实验

扩展矩阵只包含尚缺的任务：

1. 原四模型补 `TencentGR_10M_Action`：`4 × 1 × 2 = 8`；
2. UniMixer、HyFormer、UltraHSTU 跑五个数据集：`3 × 5 × 2 = 30`。

HPC01 使用一个持续的 4×L40S allocation。资源获批后，38 个任务在 allocation 内作为
串行 `srun` step 执行，不为每个训练重新排队：

```bash
mkdir -p logs
tmux new-session -d -s unirank_l40_supervisor \
  './scripts/supervise_sisa_expansion_l40.sh >> logs/unirank-sisa-l40-supervisor-launcher.log 2>&1'
tmux new-session -d -s unirank_l40_monitor \
  'MONITOR_INTERVAL_SECONDS=1800 ./scripts/monitor_sisa_expansion_l40.sh >> logs/unirank-sisa-l40-monitor-launcher.log 2>&1'
```

默认每次申请 66 小时（`2-18:00:00`，HPC01 `medium` QOS 不超过 3 天）。若集群因实际
时限、抢占或节点故障提前结束 allocation，supervisor 会在队列中没有同名申请时重新
执行 `salloc`。runner 以 `artifacts/sisa_expansion_l40/completed/task_<id>.ok` 为持久化
边界，新的 allocation 跳过已完成任务，从首个未完成任务继续。单个训练中断时不会写
完成标记，因此下次会完整重跑该任务。申请时长可用 `SISA_ALLOCATION_TIME=24:00:00`
覆盖；监控默认每 30 分钟记录队列、进度、错误、GPU 和存储状态。

数组映射：

- task `0-7`：原四模型的 TencentGR baseline/SISA；
- task `8-37`：新三模型的五数据集 baseline/SISA；
- 每个逻辑单元的偶数任务是 baseline，后一项是 SISA；
- 成功标记为 `SISA_EXPANSION_COMPLETE`。

当前申请脚本的 `account`、partition、QOS、GPU GRES、CPU、内存和时限来自 HPC01。
同步到 HPC3 后，**必须先根据 HPC3 的 Slurm 配置核对这些资源行，不能未经检查直接
启动 supervisor**。runner 在每个 allocation 开始时验证四张 GPU 均为 L40S；正式完成
仍以训练退出码与 `SISA_EXPANSION_COMPLETE` 双重门禁为准。

## 9. 环境、测试与快速运行

### 9.1 创建锁定环境

要求 Python 3.12；PyTorch 锁定为 2.8.0 + CUDA 12.6：

```bash
uv sync --locked
.venv/bin/python run_expid.py --help
```

### 9.2 CPU 回归门禁

每次修改模型、SISA、配置或 Slurm 脚本后运行：

```bash
.venv/bin/python -m unittest discover -v tests
git diff --check
bash -n scripts/*.sbatch
```

当前预期是 35 个测试全部通过。重点覆盖：

- SISA 禁用态无额外参数；
- 零 scale 精确对照；
- SISA 有限非零梯度；
- UltraHSTU Q/K 扩展与 additive bias 数学等价；
- 四卡/global-batch/seed/epoch 配置协议；
- Slurm 的 `CUDA_VISIBLE_DEVICES` 不被覆盖；
- 严格结果收集与重试映射。

### 9.3 GPU smoke

```bash
mkdir -p logs
sbatch scripts/submit_sisa_native_smoke.sbatch
```

smoke 数组覆盖七个 SISA 模型的 Taobao baseline/SISA。成功输出包含
`UNIRANK_GPU_SMOKE=...` JSON，并检查真实 batch 的 forward、backward、optimizer step、
预测有限性和 SISA 参数更新。

### 9.4 单实验入口

所有正式训练由 `run_expid.py` 和 `config/*.yaml` 驱动。SISA 变体通过 CLI 覆盖：

```bash
.venv/bin/torchrun --standalone --nproc_per_node=4 run_expid.py \
  --config ./config \
  --expid HyFormer_Taobao_Action \
  --gpu 0,1,2,3 \
  --sisa-enabled \
  --sisa-score-dim 16 \
  --sisa-lambda-init 0.1 \
  --sisa-score-scale 1.0
```

不要在 Slurm 脚本中覆盖调度器分配的 `CUDA_VISIBLE_DEVICES`。

## 10. 交接操作清单

接手或恢复工作时按以下顺序：

1. `git status --short`，确认没有来源不明的本地修改；
2. `git remote -v` 和 `git rev-parse HEAD`，确认使用个人 `origin/main`；
3. 检查五个 `datasets/<dataset_id>` 及其 block manifest/Parquet 可读性；
4. `uv sync --locked`，运行全部 CPU 测试；
5. 复核目标集群的 Slurm account、partition、QOS、GRES、内存和时限；
6. 启动单 allocation supervisor，确认只存在一个同名 L40S 申请；
7. 持续检查 error、OOM、NaN、GPU 型号和 completion marker；allocation 到期后确认自动续跑；
8. baseline/SISA 必须保持相同 GPU 型号和完整训练协议；
9. 收集结果时把严格 32-task 与扩展 38-task 分开；
10. 只提交代码、配置、测试、脚本和文档，不提交数据、日志、checkpoint 或 artifacts。

若 HPC3 checkout 有未提交工作，先创建可恢复的 backup branch 或 stash，再执行：

```bash
git fetch origin main
git merge --ff-only origin/main
```

禁止为“同步”直接删除远端工作区或覆盖数据目录。机器专用数据路径优先通过软链接解决，
不要把绝对路径写回共享配置。

## 11. 已知风险与后续工作

- 扩展 38-task 尚无正式结果；下一阶段的第一目标是 GPU smoke 和小配对门禁。
- UltraHSTU 使用 FlexAttention；CPU 只用于前向/等价性测试，正式 backward 必须在
  CUDA smoke 中验证。
- MerRec embedding 最大。扩展脚本对 MerRec 使用等价的 scalar/chunked Adagrad 路径，
  避免 PyTorch foreach 第一步的额外显存峰值。
- 单 seed 的提升不能宣称统计显著。如需显著性，应在固定调参后，仅对最强 baseline
  和候选模型使用相同的多组独立 base seed 重跑并报告检验结果。
- `artifacts/sisa_native_strict/` 和历史 Slurm 日志不在 Git 中；需要证据时应从完成
  严格实验的 HPC01 工作区重新生成或归档。

## 12. 上游、许可与引用

本分支基于 [salmon1802/UniRank](https://github.com/salmon1802/UniRank)，并受到
[FuxiCTR](https://github.com/reczoo/FuxiCTR) 的启发。代码按
[Apache License 2.0](LICENSE) 发布。

引用原 UniRank 工作：

```bibtex
@article{li2026unirank,
  title={{UniRank: Benchmarking Ranking Models for Unified Sequential Modeling and Feature Interaction}},
  author={Li, Honghao and Wang, Xianquan and Zhang, Zibin and Zhang, Yi and Lin, Kangyi and Zhang, Yiwen},
  journal={arXiv preprint arXiv:2607.19987},
  year={2026}
}
```
