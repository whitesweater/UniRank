# <消融实验名称>

状态：计划中  
负责人：<name>  
更新时间：YYYY-MM-DD

## 研究问题

说明只改变哪个因素、预期影响以及为什么值得验证。

## 控制变量

- 主实验/控制组：<对应 experiment report 和结果行>
- 模型：
- 数据集：
- 标签：
- 固定协议：GPU 数、batch、global batch、epoch、seed、精度、DDP 等
- 唯一自变量：
- 取值：

## 实验矩阵

| 状态 | 逻辑 task | 模型 | 数据集 | 变量值 | seed | job/task ID |
| --- | ---: | --- | --- | --- | ---: | --- |
| [ ] | 0 | | | | | |

## 验收标准

- 每个任务退出码为 0，并有明确完成标记；
- 指标有限，无 Traceback、OOM、NCCL 错误或 NaN；
- 与控制组使用同型号 GPU 和相同训练协议；
- 结果表包含 AUC、logloss、差值和必要的统计检验；
- `results/` 与 `figures/` 可由记录的命令重新生成。

## 结果

完成后写总体结论、按模型/数据集/标签的主要差异以及最强和最弱设置。

## 异常与重跑

记录失败 job、根因、修复方式和最终 override 映射，不覆盖或隐藏失败证据。

## 局限性

说明 seed 数量、预算、未覆盖模型/数据集以及不能据此推出的结论。

## 产物

- `configs/`：
- `results/runs.csv`：
- `results/metrics.csv`：
- `results/summary.csv`：
- `figures/`：
- 原始日志位置：

