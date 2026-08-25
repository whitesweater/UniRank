# SISA 38-task 扩展实验计划与完成记录

> 原计划日期：2026-08-24；完成日期：2026-08-25。全部 19 组配对、38 个训练元素已在
> HPC3/ACD 完成。最终结论见 [report.md](report.md)，机器可读结果见 [results/](results/)。

## 实验目标

补齐 UniRank 当前缺少的 baseline/SISA 对照实验。每个模型—数据集组合分别运行：

- baseline：模型原始实现；
- SISA：在相同模型和数据集上启用 SISA。

共需完成 19 组配对，即 38 个训练任务。

## 数据集

本轮涉及以下五个数据集：

| 数据集配置 ID | 场景 |
|---|---|
| `QK_Video_Action` | 短视频多反馈 |
| `KuaiRand_Video_Action` | 短视频多反馈 |
| `TencentGR_10M_Action` | 广告/推荐多反馈 |
| `Taobao_Action` | 电商广告多反馈 |
| `MerRec_Action` | 电商多反馈 |

## 实验矩阵（已完成）

### 原四模型补 TencentGR

这四个模型在其他四个数据集上的严格 baseline/SISA 配对已经完成，本轮只补
`TencentGR_10M_Action`。

| 状态 | 模型 | 数据集 | 需要运行 |
|---|---|---|---|
| [x] | OneTrans | `TencentGR_10M_Action` | baseline + SISA |
| [x] | HiFormer | `TencentGR_10M_Action` | baseline + SISA |
| [x] | RankMixer | `TencentGR_10M_Action` | baseline + SISA |
| [x] | Zenith | `TencentGR_10M_Action` | baseline + SISA |

小计：4 组配对，8 个训练任务。

### 新三模型运行五个数据集

UniMixer、HyFormer 和 UltraHSTU 分别在全部五个数据集上运行 baseline/SISA 配对。

| 状态 | 模型 | 数据集 | 需要运行 |
|---|---|---|---|
| [x] | UniMixer | `QK_Video_Action` | baseline + SISA |
| [x] | UniMixer | `KuaiRand_Video_Action` | baseline + SISA |
| [x] | UniMixer | `TencentGR_10M_Action` | baseline + SISA |
| [x] | UniMixer | `Taobao_Action` | baseline + SISA |
| [x] | UniMixer | `MerRec_Action` | baseline + SISA |
| [x] | HyFormer | `QK_Video_Action` | baseline + SISA |
| [x] | HyFormer | `KuaiRand_Video_Action` | baseline + SISA |
| [x] | HyFormer | `TencentGR_10M_Action` | baseline + SISA |
| [x] | HyFormer | `Taobao_Action` | baseline + SISA |
| [x] | HyFormer | `MerRec_Action` | baseline + SISA |
| [x] | UltraHSTU | `QK_Video_Action` | baseline + SISA |
| [x] | UltraHSTU | `KuaiRand_Video_Action` | baseline + SISA |
| [x] | UltraHSTU | `TencentGR_10M_Action` | baseline + SISA |
| [x] | UltraHSTU | `Taobao_Action` | baseline + SISA |
| [x] | UltraHSTU | `MerRec_Action` | baseline + SISA |

小计：15 组配对，30 个训练任务。

## 数量核对

| 部分 | 模型数 | 数据集数 | 配对数 | 训练任务数 |
|---|---:|---:|---:|---:|
| 原四模型补 TencentGR | 4 | 1 | 4 | 8 |
| 新三模型跑五个数据集 | 3 | 5 | 15 | 30 |
| 合计 | 7 | 5（去重） | 19 | 38 |

## 完成与结果整理

全部组合均已满足 baseline 和 SISA 两个训练成功并取得最终测试指标的条件。结果已按
“模型、数据集、标签、baseline 指标、SISA 指标、差值”记录在
[results/paired_summary.csv](results/paired_summary.csv)，并与 32-task 严格实验分开归档。
