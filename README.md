<p align="center">
  <img src="./assets/figures/unirank_logo.png" alt="UniRank logo" width="720">
</p>

# UniRank 项目说明与实验交接

> 最后更新：2026-09-03。本仓库是 UniRank 的可复现研究分支，同时包含 SISA
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
| TencentGR 补充矩阵 | 已完成并审计 | 原四模型 × TencentGR × baseline/SISA，共 8 个任务 |
| 新三模型 SISA | 已完成代码与测试 | UniMixer、HyFormer、UltraHSTU |
| 新三模型五数据集矩阵 | 已完成并审计 | 3 模型 × 5 数据集 × baseline/SISA，共 30 个任务 |
| HPC3/ACD 扩展矩阵 | 已完成 | 38 个独立 array element；每个元素使用 4×H100；38/38 有效 |
| 本地回归测试 | 已通过 | 当前 85 个测试全部通过 |

最重要的交接边界：**32 个原严格任务和 38 个 HPC3/ACD 扩展任务均已完成并审计。**
后续新增工作应作为独立消融实验归档，不要覆盖这两组正式结果。

## 3. 仓库与运行环境

| 位置 | 路径/地址 | 用途 |
|---|---|---|
| GitHub | `https://github.com/whitesweater/UniRank` | 公开主远端，`main` 是同步源 |
| 上游 GitHub | `https://github.com/salmon1802/UniRank` | 只用于跟踪官方更新，远端名 `upstream` |

集群 checkout、SSH 别名和数据挂载点属于部署配置，不写入版本库。历史实验报告仅保留复现
结论所需的硬件、协议和 job/task 映射。

Git 只同步实现和复现必需文件。以下内容按设计不进入仓库：

- `datasets/`：预处理数据和软链接；
- `logs/`、`checkpoints/`：Slurm 输出、训练指标日志和本机保存的最佳模型权重；
- `artifacts/`：阶段性审计结果、失败现场和其他中间产物；
- `.venv/`、编译缓存和临时目录；
- 本机专用的中间文件。

因此，“代码三端一致”不等于“数据和历史日志已经复制”。在新的集群运行前必须单独
核对数据路径、Slurm 资源、Python/CUDA 环境和输出目录。

## 4. 目录和关键入口

```text
UniRank/
├── .git/                            # Git 历史、分支和对象数据库
├── .venv/                           # uv 创建的本机 Python/CUDA 依赖环境
├── assets/                          # README 使用的 Logo 和流程图
├── config/
│   ├── dataset_config.yaml          # 数据路径、schema、label、vocab
│   └── model_config.yaml            # 模型/数据集实验配置
├── model_zoo/                       # 15 个模型及 SISA 适配位置
├── unirank/                         # dataloader、训练、指标和公共层
├── data/                            # 数据下载、预处理和统计脚本，不存训练数据
├── datasets/                        # 实际训练使用的 blocked Parquet 数据
├── benchmark/                       # 上游 UniRank 参考日志和论文基线
├── checkpoints/                     # 训练指标日志和默认保留的最佳模型权重
├── logs/                            # Slurm 作业的原始 stdout/stderr
├── artifacts/                       # 阶段性审计结果和失败重试现场
├── experiments/                     # 实验计划、报告、结果表与消融模板
│   ├── README.md                    # 实验总索引与归档规范
│   ├── sisa_native_strict/          # 已完成 32-task 严格实验
│   │   ├── report.md                # 统一正式报告
│   │   ├── results/                 # 小型机器可读结果
│   │   └── migration/               # HPC01 原始报告与迁移证据包
│   ├── sisa_expansion_acd/          # 已完成 38-task HPC3/ACD 扩展实验
│   ├── sisa_single_seed20262028/     # seed 20262028 结果与两-seed 比较
│   ├── sisa_single_seed20262029/     # seed 20262029 结果与同协议比较
│   ├── sisa_three_seed_unified/      # 三-seed、baseline 与论文表统一分析
│   ├── ablations/                   # 后续消融实验索引
│   └── templates/ablation/          # 消融报告模板
├── scripts/
│   ├── submit_sisa_native_strict.sbatch
│   ├── submit_sisa_expansion.sbatch
│   ├── submit_sisa_expansion_acd.sbatch
│   ├── collect_sisa_expansion_acd_results.py
│   ├── audit_sisa_expansion_baselines.py
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
├── pyproject.toml / uv.lock         # 锁定环境
└── README.md                        # 项目说明与主交接文档
```

各目录的职责和保留策略如下：

| 目录 | 职责 | 是否可重新生成/清理 |
|---|---|---|
| `.git/` | 保存 Git 提交历史、分支和版本对象 | 不可清理；删除后当前目录将失去 Git 历史 |
| `.venv/` | Python 3.12、PyTorch 2.8/CUDA 12.6 及项目依赖 | 可用 `uv sync --locked` 重建；近期还要实验时保留 |
| `assets/` | README 中引用的项目 Logo、训练和测试流程图 | 文档依赖，保留 |
| `config/` | 数据集 schema、标签、路径以及模型超参数 | 训练和复现必需，保留 |
| `unirank/` | UniRank 核心框架：特征、数据加载、训练、指标和公共层 | 核心源码，保留 |
| `model_zoo/` | 15 个模型实现及各模型的 SISA 接入代码 | 核心源码，保留 |
| `data/` | 下载、预处理、转换和统计数据集的程序 | 这是脚本而不是数据，保留 |
| `datasets/` | 五个数据集的实际训练数据和分块元数据 | 原始运行依赖；体积大但后续实验需要，保留 |
| `benchmark/` | 上游 UniRank 的模型/数据集参考日志 | baseline AUC 复现审计依赖，保留 |
| `checkpoints/` | 每次训练的 `.log`、验证集最佳权重和非最佳权重归档 | 最佳 `.model` 留在数据集目录；其他 checkpoint 软删除到 `archive/` |
| `logs/` | Slurm array、预检、gate、重试和训练的原始输出 | 正式任务日志应保留；失败或废弃日志可归档压缩 |
| `artifacts/` | 阶段性 baseline 审计 CSV 和失败任务现场 | 最终结果入档后可精简重复内容，但目录供后续任务继续写入 |
| `experiments/` | 正式实验计划、报告、最终机器可读结果和消融模板 | 实验结论的唯一归档入口，保留并提交 Git |
| `scripts/` | Slurm 提交、环境预检、监控、结果收集和审计工具 | 实验复现和后续消融需要，保留 |
| `tests/` | SISA、数据协议、Slurm 运行和结果审计回归测试 | 修改代码后的验证依据，保留 |

运行产物之间的边界是：`benchmark/` 保存上游参考答案；`checkpoints/` 保存本次训练的
指标日志和最佳模型权重；`logs/` 保存 Slurm 原始输出；`artifacts/` 保存中间审计
和失败现场；`experiments/` 保存整理完成、可以引用的正式结果。

### 4.1 模型权重的保存与删除

`config/model_config.yaml` 将 `model_root` 设为 `./checkpoints/`。训练过程中，rank 0 会把
验证集指标最好的模型写到：

```text
checkpoints/<dataset_id>/<run_id>.model
```

权重内容是 PyTorch `state_dict`。每次 validation 都会产生一份 checkpoint：当前最佳
模型保留在上述固定路径；非最佳模型以及被新最佳模型替换的旧最佳模型执行“软删除”，
移动到按 run 和训练 session 隔离的归档目录：

```text
checkpoints/<dataset_id>/archive/<run_id>/<session_id>/
├── preexisting_best.model
└── eval_<序号>_epoch_<epoch>_step_<step>.model
```

训练结束后程序重新加载主目录中的最佳权重完成测试集评测，并继续保留该 `.model`，供
后续推理、复核或微调。代码不再提供测试后硬删除 checkpoint 的命令行路径；即使同一
`run_id` 重新训练，原有最佳权重也会先进入新 session 的归档目录，不会被直接覆盖。

当前已完成的 38 个扩展训练是在旧删除逻辑下运行的，因此没有留下可恢复的 `.model`；
`checkpoints/` 中现存的 38 个文件全部是 `.log` 训练和测试指标记录。代码更新之后的新
实验会同时留下 `.log` 与 `.model`。权重属于本机运行产物，受 `.gitignore` 中的
`*.model` 规则保护，不会被误提交到 Git。

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
生成方式见 [严格实验报告](experiments/sisa_native_strict/report.md)和
[机器可读结果](experiments/sisa_native_strict/results/)。HPC01 原始证据已按
[迁移清单](experiments/sisa_native_strict/migration/migration_files.txt)完成 checksum 迁移。
历史 pilot 为
单卡 global batch 8192，只作诊断，禁止混入严格结果表。

## 8. 已完成的 38-task HPC3/ACD 扩展实验

扩展矩阵已经在 HPC3 `acd_u` 队列完成：

1. 原四模型补 `TencentGR_10M_Action`：`4 × 1 × 2 = 8`；
2. UniMixer、HyFormer、UltraHSTU 跑五个数据集：`3 × 5 × 2 = 30`。

每个训练是独立 Slurm array element，使用 4×H100 80GB；最多四个元素并发。统一协议
仍为每卡 batch 8192、global batch 32768、accumulation 1、一个 epoch、seed 20262027、
BF16 和 blocked DDP。

最终结果：

- 有效训练元素 `38/38`，GPU 配对和指标配对错误均为 0；
- 19 个 baseline、71 个标签全部通过与 UniRank benchmark 的 AUC 绝对偏差 `≤0.01`
  门槛，最大偏差为 `0.009366`；
- 71 个 baseline/SISA 标签对的平均 ΔAUC 为 `+0.003802`，其中 48 个提升、23 个下降；
- 完整恢复了 task 0 的 TencentGR 序列池化、task 12/13 的官方 UniMixer 参数，以及
  task 29/35/37 的 UltraHSTU FlexAttention 编译限制。

统一档案入口：

- [最终报告](experiments/sisa_expansion_acd/report.md)
- [机器可读结果](experiments/sisa_expansion_acd/results/)
- [原始实验计划与完成矩阵](experiments/sisa_expansion_acd/planning.md)

后续消融实验统一放在 [experiments/ablations/](experiments/ablations/README.md)，使用
[消融模板](experiments/templates/ablation/README.md)，不再在仓库根目录新增零散报告。

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
9. 收集结果时把严格 32-task、扩展 38-task 和后续消融 study 分开；
10. 小型结果表与报告归档到 `experiments/`；不提交数据、完整日志、checkpoint 或运行时 artifacts。

若 HPC3 checkout 有未提交工作，先创建可恢复的 backup branch 或 stash，再执行：

```bash
git fetch origin main
git merge --ff-only origin/main
```

禁止为“同步”直接删除远端工作区或覆盖数据目录。机器专用数据路径优先通过软链接解决，
不要把绝对路径写回共享配置。

## 11. 已知风险与后续工作

- 扩展 38-task 已完成；下一阶段应按 `experiments/ablations/` 的单变量规范设计消融，
  不应直接复用临时 job ID 或散落的 stdout 作为报告。
- UltraHSTU 使用 FlexAttention；CPU 只用于前向/等价性测试，正式 backward 必须在
  CUDA smoke 中验证。
- MerRec embedding 最大。扩展脚本对 MerRec 使用等价的 scalar/chunked Adagrad 路径，
  避免 PyTorch foreach 第一步的额外显存峰值。
- 单 seed 的提升不能宣称统计显著。如需显著性，应在固定调参后，仅对最强 baseline
  和候选模型使用相同的多组独立 base seed 重跑并报告检验结果。
- 32-task 严格实验的 32 份 Slurm 日志、32 份训练指标日志和原始 CSV 已从 HPC01
  checksum 迁移到当前 HPC3；运行时副本仍受 `.gitignore` 保护，小型结果表已归档到
  `experiments/sisa_native_strict/results/`。

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
