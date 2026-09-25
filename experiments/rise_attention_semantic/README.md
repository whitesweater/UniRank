# Attention语义案例

先读 `report.md`。本研究使用原有四个数据集中的KuaiRand，为已有RankMixer–RISE补充真实视频文案与局部修正开关的预测比较，没有新增数据集或训练。

- 正文图：`figures/semantic_improved`、`figures/semantic_degraded`，PDF/SVG/PNG。
- 完整文字与数值：`figures/` 中的CSV与选择receipt。
- 固定协议：`protocol.md`；同源内容映射依据：`metadata_audit.md`。
- 数值结果与来源：`results/`；复核记录：`audit/`。

成功推理作业646410。两个案例真实反馈均为点击：改善例p(click)49.80%→53.91%，退化例60.16%→55.47%。它们说明固定模型的局部敏感性，不是独立Baseline对比或总体AUC提升。
