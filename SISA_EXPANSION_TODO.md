# SISA 扩展实验 TODO

> 更新日期：2026-08-24。本文件只记录需要完成的模型与数据集组合，不包含任何机器、
> GPU 或集群调度细节。

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

## 待完成矩阵

### 原四模型补 TencentGR

这四个模型在其他四个数据集上的严格 baseline/SISA 配对已经完成，本轮只补
`TencentGR_10M_Action`。

| 状态 | 模型 | 数据集 | 需要运行 |
|---|---|---|---|
| [ ] | OneTrans | `TencentGR_10M_Action` | baseline + SISA |
| [ ] | HiFormer | `TencentGR_10M_Action` | baseline + SISA |
| [ ] | RankMixer | `TencentGR_10M_Action` | baseline + SISA |
| [ ] | Zenith | `TencentGR_10M_Action` | baseline + SISA |

小计：4 组配对，8 个训练任务。

### 新三模型运行五个数据集

UniMixer、HyFormer 和 UltraHSTU 分别在全部五个数据集上运行 baseline/SISA 配对。

| 状态 | 模型 | 数据集 | 需要运行 |
|---|---|---|---|
| [ ] | UniMixer | `QK_Video_Action` | baseline + SISA |
| [ ] | UniMixer | `KuaiRand_Video_Action` | baseline + SISA |
| [ ] | UniMixer | `TencentGR_10M_Action` | baseline + SISA |
| [ ] | UniMixer | `Taobao_Action` | baseline + SISA |
| [ ] | UniMixer | `MerRec_Action` | baseline + SISA |
| [ ] | HyFormer | `QK_Video_Action` | baseline + SISA |
| [ ] | HyFormer | `KuaiRand_Video_Action` | baseline + SISA |
| [ ] | HyFormer | `TencentGR_10M_Action` | baseline + SISA |
| [ ] | HyFormer | `Taobao_Action` | baseline + SISA |
| [ ] | HyFormer | `MerRec_Action` | baseline + SISA |
| [ ] | UltraHSTU | `QK_Video_Action` | baseline + SISA |
| [ ] | UltraHSTU | `KuaiRand_Video_Action` | baseline + SISA |
| [ ] | UltraHSTU | `TencentGR_10M_Action` | baseline + SISA |
| [ ] | UltraHSTU | `Taobao_Action` | baseline + SISA |
| [ ] | UltraHSTU | `MerRec_Action` | baseline + SISA |

小计：15 组配对，30 个训练任务。

## 数量核对

| 部分 | 模型数 | 数据集数 | 配对数 | 训练任务数 |
|---|---:|---:|---:|---:|
| 原四模型补 TencentGR | 4 | 1 | 4 | 8 |
| 新三模型跑五个数据集 | 3 | 5 | 15 | 30 |
| 合计 | 7 | 5（去重） | 19 | 38 |

## 完成与结果整理

只有 baseline 和 SISA 两个训练都成功并取得最终测试指标后，才勾选对应组合。结果整理
时按“模型、数据集、标签、baseline 指标、SISA 指标、差值”记录，并与已经完成的
32-task 严格实验分开汇总。
