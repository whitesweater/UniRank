# RISE Attention Redistribution

HiFormer–TencentGR 的既有checkpoint推理诊断已完成。首先阅读 `report.md`。

- `protocol.md`：固定采样、选择及数值门槛，以及失败恢复说明。
- `figures/`：正文案例、TV分布、P10/P90附录案例，均有PDF/SVG/PNG；另含完整绘图数值和选择凭据。
- `results/`：512条交互、474位用户的compact统计与独立核验。
- `audit/`：失败尝试、CPU测试和视觉QA记录。

成功作业646303；平均TV=0.164466，首位改变401/512。采集前后及与历史正式预测均逐位相等。没有新增训练；“修正前”不等于独立Baseline。

原始数组、完整批次缓存和冻结源码保留于 `artifacts/rise_attention_redistribution/tencent_646303/`。采集、审计和绘图脚本位于根目录 `scripts/`，复现命令见报告。
